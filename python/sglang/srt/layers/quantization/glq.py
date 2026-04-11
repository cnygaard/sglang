"""GLQ (E8 Lattice Quantization) support for sglang.

Ported from glq_vllm/ for sglang integration. Reuses the existing
glq.inference_kernel CUDA extension (torch cpp_extension) so no kernel
code is duplicated into sglang's JIT kernel tree.

Weights stay compressed in GPU memory as int16 codebook indices +
fp16 sign vectors. Dequantization runs on-the-fly during matmul via
fused CUDA kernels: input RHT → dequant+matvec → output RHT.

Usage:
    python -m sglang.launch_server \\
        --model xv0y5ncu/SmolLM3-3B-GLQ-3.5bpw --quantization glq
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.parameter import BasevLLMParameter
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)

# Runtime dependency: the glq package supplies the CUDA kernels and codebook
from glq import inference_kernel as _ik
from glq.codebook import E8ShellCodebook
from glq.inference_kernel import glq_dequant_matmul, _try_load_cuda_ext


# ──────────────────────────────────────────────────────────────────────────
# Dense dequantization (for Mamba layers that can't use compressed kernels)
# ──────────────────────────────────────────────────────────────────────────

def _dequantize_glq_weight(Qidxs, SU, SV, Wscale, codebook,
                           Qidxs2=None, inv_resid_scale=0.0, codebook2=None,
                           out_features=None, in_features=None):
    """Dequantize GLQ indices to a dense fp16 weight matrix (CPU)."""
    from glq.hadamard import _pytorch_fht as fast_hadamard_transform
    m_pad, n_blocks = Qidxs.shape
    n_pad = n_blocks * 8
    W_rht = codebook.decode(Qidxs.long().reshape(-1)).reshape(m_pad, n_pad).float()
    if Qidxs2 is not None and inv_resid_scale != 0.0 and codebook2 is not None:
        W_rht2 = codebook2.decode(Qidxs2.long().reshape(-1)).reshape(m_pad, n_pad).float()
        W_rht = W_rht + W_rht2 * inv_resid_scale
    W_rht = W_rht * Wscale.float()
    W = fast_hadamard_transform(W_rht.clone())
    W = W * SV.float().unsqueeze(0)
    W = fast_hadamard_transform(W.T.clone()).T
    W = W * SU.float().unsqueeze(1)
    if out_features is not None and in_features is not None:
        W = W[:out_features, :in_features]
    return W.half()


# ──────────────────────────────────────────────────────────────────────────
# Shared codebook singleton (lazy-loaded on first layer)
# ──────────────────────────────────────────────────────────────────────────

_codebook = None
_codebook2_small = None
_codebook_device = None


def _ensure_codebook(device, max_bpw: int = 2):
    """Lazy-load the E8 codebook and move to target device. Upgrade cb2 for 3/4bpw."""
    global _codebook, _codebook2_small, _codebook_device

    if _codebook is not None and _codebook_device == device:
        if max_bpw >= 3 and _codebook2_small is None:
            _codebook2_small = _codebook
        return _codebook, _codebook2_small

    cb_path = os.path.join(
        os.path.dirname(_ik.__file__), "e8_codebook.pt"
    )
    if os.path.exists(cb_path):
        cb = E8ShellCodebook.load(cb_path, device="cpu")
    else:
        cb = E8ShellCodebook(device="cpu", verbose=False)

    cb2 = cb if max_bpw >= 3 else None
    cb._move_to_device(device)
    if cb2 is not None and cb2 is not cb:
        cb2._move_to_device(device)

    _codebook = cb
    _codebook2_small = cb2
    _codebook_device = device
    return cb, cb2


# ──────────────────────────────────────────────────────────────────────────
# GLQ buffer helpers
# ──────────────────────────────────────────────────────────────────────────

def _glq_pad(n: int) -> int:
    """Next power of 2."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _glq_weight_loader(param, loaded_weight, *args, **kwargs):
    """Default GLQ weight loader — handles standard and per-expert params."""
    expert_id = kwargs.get("expert_id")
    if expert_id is not None:
        if param.data.dim() >= 2:
            slot = param.data[expert_id]
            if slot.shape != loaded_weight.shape:
                new_shape = list(param.data.shape)
                for d in range(loaded_weight.dim()):
                    new_shape[d + 1] = loaded_weight.shape[d]
                param.data = torch.zeros(
                    new_shape, dtype=param.data.dtype, device=param.data.device
                )
            param.data[expert_id].copy_(loaded_weight)
        elif param.data.dim() == 1 and loaded_weight.dim() == 0:
            param.data[expert_id] = loaded_weight.item()
        elif param.data.dim() == 1 and loaded_weight.dim() >= 1:
            if expert_id == 0:
                if param.data.shape != loaded_weight.shape:
                    param.data = torch.empty_like(loaded_weight)
                param.data.copy_(loaded_weight)
        elif param.data.dim() == 0:
            param.data.copy_(loaded_weight)
        return True
    if param.data.shape != loaded_weight.shape:
        param.data = torch.empty_like(loaded_weight)
    param.data.copy_(loaded_weight)
    return True


def _make_glq_param(tensor: torch.Tensor) -> torch.nn.Parameter:
    """Create nn.Parameter with GLQ weight_loader attached."""
    p = torch.nn.Parameter(tensor, requires_grad=False)
    p.weight_loader = _glq_weight_loader
    return p


def _register_glq_buffers(layer, prefix: str, out_size: int, in_size: int):
    """Register one set of GLQ compressed buffers on the layer."""
    m_pad = _glq_pad(out_size)
    n_pad = _glq_pad(in_size)
    n_blocks = n_pad // 8
    p = prefix
    setattr(layer, f"Qidxs{p}", _make_glq_param(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16)))
    setattr(layer, f"SU{p}", _make_glq_param(
        torch.ones(m_pad, dtype=torch.float16)))
    setattr(layer, f"SV{p}", _make_glq_param(
        torch.ones(n_pad, dtype=torch.float16)))
    setattr(layer, f"Wscale{p}", _make_glq_param(
        torch.ones((), dtype=torch.float32)))
    setattr(layer, f"Qidxs2{p}", _make_glq_param(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16)))
    setattr(layer, f"inv_resid_scale{p}", _make_glq_param(
        torch.zeros((), dtype=torch.float32)))
    return m_pad, n_pad


# ──────────────────────────────────────────────────────────────────────────
# Per-shard parameter for fused QKV / gate-up linear layers
# ──────────────────────────────────────────────────────────────────────────

class GLQShardedParameter(BasevLLMParameter):
    """Parameter holding per-shard GLQ buffers for fused QKV / gate-up layers.

    sglang's stacked_params_mapping routes q_proj.Qidxs → qkv_proj.Qidxs with
    shard_id="q"|"k"|"v"; this class stores each shard in a separate internal
    buffer because each shard is padded independently to a power-of-2.
    """

    def __new__(cls, shard_sizes, inner_dim, dtype, **kwargs):
        total_m = sum(_glq_pad(s) if s > 1 else 1 for s in shard_sizes)
        if inner_dim > 0:
            data = torch.zeros(total_m, inner_dim, dtype=dtype)
        elif inner_dim == -1:
            # 1D vector per shard (SU sign vector)
            data = torch.zeros(total_m, dtype=dtype)
        else:
            data = torch.zeros(len(shard_sizes), dtype=dtype)
        return super().__new__(cls, data=data, **kwargs)

    def __init__(
        self,
        shard_sizes: List[int],
        inner_dim: int,
        dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        total_m = sum(_glq_pad(s) if s > 1 else 1 for s in shard_sizes)
        if inner_dim > 0:
            data = torch.zeros(total_m, inner_dim, dtype=dtype)
        elif inner_dim == -1:
            data = torch.zeros(total_m, dtype=dtype)
        else:
            data = torch.zeros(len(shard_sizes), dtype=dtype)
        super().__init__(data=data, weight_loader=weight_loader)
        self._shard_sizes = shard_sizes
        self._inner_dim = inner_dim
        self._dtype = dtype

        # Allocate per-shard buffers with power-of-2 padding
        self._shard_data = []
        for sz in shard_sizes:
            m_pad = _glq_pad(sz) if sz > 1 else 1
            if inner_dim > 0:
                self._shard_data.append(
                    torch.zeros(m_pad, inner_dim, dtype=dtype))
            elif inner_dim == -1:
                self._shard_data.append(torch.ones(m_pad, dtype=dtype))
            else:
                self._shard_data.append(torch.zeros((), dtype=dtype))

    @staticmethod
    def _shard_id_as_int(shard_id) -> int:
        if isinstance(shard_id, int):
            return shard_id
        return {"q": 0, "k": 1, "v": 2}.get(shard_id, 0)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        idx = self._shard_id_as_int(shard_id)
        if idx < len(self._shard_data):
            if self._shard_data[idx].shape != loaded_weight.shape:
                self._shard_data[idx] = torch.empty_like(loaded_weight)
            self._shard_data[idx].copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        idx = self._shard_id_as_int(shard_id) if shard_id is not None else 0
        if idx < len(self._shard_data):
            if self._shard_data[idx].shape != loaded_weight.shape:
                self._shard_data[idx] = torch.empty_like(loaded_weight)
            self._shard_data[idx].copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        if len(self._shard_data) == 1:
            self._shard_data[0].copy_(loaded_weight)
        else:
            # Unsplit fused weight (e.g. Mamba in_proj stored as single tensor).
            # Stash for dequantization in process_weights_after_loading.
            self._unsplit_weight = loaded_weight.clone()

    @property
    def weight_loader(self) -> Callable:
        """Override to return our shard router (not the layer's weight_loader)."""
        return self._glq_shard_loader

    def _glq_shard_loader(self, param, loaded_weight, *args, **kwargs):
        shard_id = kwargs.get("shard_id") or (args[0] if args else None)
        if shard_id is not None:
            remaining = {k: v for k, v in kwargs.items() if k != "shard_id"}
            self.load_qkv_weight(loaded_weight, shard_id=shard_id, **remaining)
        else:
            self.load_column_parallel_weight(loaded_weight)

    def get_shard(self, idx: int) -> torch.Tensor:
        return self._shard_data[idx]

    @property
    def num_shards(self) -> int:
        return len(self._shard_data)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result._shard_data = [s.to(*args, **kwargs) for s in result._shard_data]
        return result

    def cuda(self, device=None):
        result = super().cuda(device)
        result._shard_data = [s.cuda(device) for s in result._shard_data]
        return result


# ──────────────────────────────────────────────────────────────────────────
# Forward pass: input RHT → dequant+matmul → output RHT
# ──────────────────────────────────────────────────────────────────────────

_VLLM_USE_TRITON = False  # Prefer CUDA C kernels (1.3-1.6× faster than Triton)


def _glq_apply_shard(
    x, device, cb, cb2, Qidxs, SU, SV, wscale,
    has_stage2, inv_rs, Qidxs2, out_features, in_features,
    m_pad, n_pad, log_n, log_m,
):
    """Full GLQ forward for one set of weight buffers.

    Uses CUDA C kernels from glq.inference_kernel._glq_cuda when available,
    falling back to Triton for n_pad > 16384.
    """
    dtype = x.dtype
    B = x.shape[0]
    _use_cuda_c = _ik._glq_cuda is not None and not _VLLM_USE_TRITON

    # Safety net: ensure weight tensors on the right device (first forward call)
    if Qidxs.device != x.device:
        Qidxs = Qidxs.to(device)
    if SU.device != x.device:
        SU = SU.to(device)
    if SV.device != x.device:
        SV = SV.to(device)
    if Qidxs2 is not None and Qidxs2.device != x.device:
        Qidxs2 = Qidxs2.to(device)

    # Input RHT: x (B,in_features) → x_rht (B,n_pad) fp32
    x_rht = torch.empty(B, n_pad, dtype=torch.float32, device=device)
    rsqrt_n = 1.0 / math.sqrt(n_pad)
    if n_pad <= 16384 and _use_cuda_c:
        _ik._glq_cuda.glq_input_rht_cuda(
            x.half().contiguous(), SV, x_rht,
            in_features, in_features, rsqrt_n, n_pad, log_n,
        )
    else:
        from glq.quantized_linear import _input_rht_kernel
        _input_rht_kernel[(B,)](
            x, SV, x_rht, in_features, x.stride(0),
            rsqrt_n, N=n_pad, LOG_N=log_n, num_warps=8,
        )

    # Dequant + matmul: x_rht (B,n_pad) → y_rht (B,m_pad) fp32
    cb_packed = getattr(cb, "codebook_packed", None)
    cb2_half = cb2.codebook_half if has_stage2 and cb2 is not None else None

    y_rht = glq_dequant_matmul(
        x_rht, Qidxs, cb.codebook_half, wscale,
        Qidxs2=Qidxs2, codebook2=cb2_half,
        inv_resid_scale=inv_rs, codebook_packed=cb_packed,
    )

    # Output RHT: y_rht (B,m_pad) → y (B,out_features)
    rsqrt_m = 1.0 / math.sqrt(m_pad)
    if m_pad <= 16384 and _use_cuda_c:
        y = torch.empty(B, out_features, dtype=torch.float16, device=device)
        _ik._glq_cuda.glq_output_rht_cuda(
            y_rht, SU, y, out_features, m_pad, log_m, rsqrt_m,
        )
        if dtype != torch.float16:
            y = y.to(dtype)
    else:
        from glq.quantized_linear import _output_rht_kernel
        output_fp16 = dtype == torch.float16
        y = torch.empty(B, out_features, dtype=dtype, device=device)
        _output_rht_kernel[(B,)](
            y_rht, SU, y, out_features, y_rht.stride(0), y.stride(0),
            rsqrt_m, OUTPUT_FP16=output_fp16, M=m_pad, LOG_M=log_m, num_warps=8,
        )
    return y


def _glq_apply_single(x, layer, prefix, cb, cb2, device):
    """Apply one set of GLQ buffers stored on the layer under `{name}{prefix}`."""
    Qidxs = getattr(layer, f"Qidxs{prefix}").to(device)
    SU = getattr(layer, f"SU{prefix}").to(device)
    SV = getattr(layer, f"SV{prefix}").to(device)
    wscale = getattr(layer, f"_glq_wscale{prefix}")
    has_stage2 = getattr(layer, f"_glq_has_stage2{prefix}")
    inv_rs = getattr(layer, f"_glq_inv_rs{prefix}")
    m_pad = getattr(layer, f"_glq_m_pad{prefix}")
    n_pad = getattr(layer, f"_glq_n_pad{prefix}")
    log_n = getattr(layer, f"_glq_log_n{prefix}")
    log_m = getattr(layer, f"_glq_log_m{prefix}")
    out_features = getattr(layer, f"_glq_out{prefix}")
    in_features = getattr(layer, f"_glq_in{prefix}")

    Qidxs2 = getattr(layer, f"Qidxs2{prefix}").to(device) if has_stage2 else None
    return _glq_apply_shard(
        x, device, cb, cb2,
        Qidxs=Qidxs, SU=SU, SV=SV, wscale=wscale,
        has_stage2=has_stage2, inv_rs=inv_rs, Qidxs2=Qidxs2,
        out_features=out_features, in_features=in_features,
        m_pad=m_pad, n_pad=n_pad, log_n=log_n, log_m=log_m,
    )


# ──────────────────────────────────────────────────────────────────────────
# GLQ Quantization Config
# ──────────────────────────────────────────────────────────────────────────

class GLQConfig(QuantizationConfig):
    """sglang quantization config for GLQ (E8 lattice codebook + RHT)."""

    def __init__(self, bpw: int = 2, layer_bpw: Optional[Dict[str, int]] = None):
        super().__init__()
        self.bpw = bpw
        self.layer_bpw = layer_bpw or {}

    def __repr__(self) -> str:
        return f"GLQConfig(bpw={self.bpw})"

    def get_name(self) -> str:
        return "glq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Volta+

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quantize_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GLQConfig":
        return cls(
            bpw=config.get("bpw", 2),
            layer_bpw=config.get("layer_bpw", None),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            bpw = self.layer_bpw.get(prefix, self.bpw)
            return GLQLinearMethod(self, bpw=bpw)
        return None


# ──────────────────────────────────────────────────────────────────────────
# GLQ Linear Method
# ──────────────────────────────────────────────────────────────────────────

class GLQLinearMethod(LinearMethodBase):
    """GLQ fused dequant+matmul linear method.

    Compressed weights (int16 codebook indices + fp16 sign vectors) stay in
    GPU memory. Dequantization runs on-the-fly during matmul via fused CUDA
    kernels from the glq.inference_kernel extension module.
    """

    def __init__(self, quant_config: GLQConfig, bpw: int = 2):
        self.quant_config = quant_config
        self.bpw = bpw

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        is_fused = len(output_partition_sizes) > 1
        layer.glq_is_fused = is_fused
        layer.glq_in_features = input_size_per_partition
        layer.glq_bpw = self.bpw

        weight_loader = extra_weight_attrs.get("weight_loader")

        if is_fused:
            # Fused QKV / gate-up: per-shard GLQ buffers via GLQShardedParameter
            layer.glq_shard_sizes = output_partition_sizes
            layer.glq_num_shards = len(output_partition_sizes)
            n_pad = _glq_pad(input_size_per_partition)
            n_blocks = n_pad // 8

            layer.Qidxs = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16,
                weight_loader=weight_loader,
            )
            # SU is a 1D sign vector per shard — use -1 sentinel for 1D alloc
            # (fix for Mamba in_proj fused projections; see glq commit 299f00b)
            layer.SU = GLQShardedParameter(
                output_partition_sizes, -1, torch.float16,
                weight_loader=weight_loader,
            )
            # SV is per-shard too: the quantizer uses an independent random
            # Hadamard rotation per layer, so q/k/v have different SV vectors
            # even though they share the input dim. Each shard stores a full
            # n_pad-length SV (not m_pad-length); reuse GLQShardedParameter
            # with shard_sizes=[n_pad]*num_shards to get that layout.
            layer.SV = GLQShardedParameter(
                [n_pad] * len(output_partition_sizes), -1, torch.float16,
                weight_loader=weight_loader,
            )
            layer.Wscale = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32,
                weight_loader=weight_loader,
            )
            layer.Qidxs2 = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16,
                weight_loader=weight_loader,
            )
            layer.inv_resid_scale = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32,
                weight_loader=weight_loader,
            )

            layer.glq_n_pad = n_pad

        # Dummy weight param to satisfy modules that access `layer.weight`
        if not hasattr(layer, "weight"):
            layer.weight = _make_glq_param(torch.empty(1, dtype=params_dtype))

        if not is_fused:
            out_sz = sum(output_partition_sizes)
            layer.glq_out_features = out_sz
            m_pad, n_pad = _register_glq_buffers(
                layer, "", out_sz, input_size_per_partition,
            )
            layer.glq_m_pad = m_pad
            layer.glq_n_pad = n_pad

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Ensure codebook on device, cache scalars for fast apply()."""
        device = next(layer.parameters()).device
        bpw = getattr(layer, "glq_bpw", 2)

        # Unsplit fused layers (e.g. Mamba in_proj): checkpoint stores a
        # single GLQ tensor but the layer has multi-shard buffers.
        # Dequantize to dense fp16 and use F.linear at inference time.
        if (getattr(layer, "glq_is_fused", False)
                and hasattr(layer.Qidxs, "_unsplit_weight")):
            cb = E8ShellCodebook(device="cpu", verbose=False)
            q = layer.Qidxs._unsplit_weight.cpu()
            su = layer.SU._unsplit_weight.cpu()
            sv = layer.SV._unsplit_weight.cpu()
            ws = layer.Wscale._unsplit_weight.cpu()
            inv_rs_t = getattr(layer.inv_resid_scale, "_unsplit_weight", None)
            inv_rs = inv_rs_t.cpu().item() if inv_rs_t is not None and inv_rs_t.numel() == 1 else 0.0
            q2 = layer.Qidxs2._unsplit_weight.cpu() if (
                inv_rs != 0.0 and hasattr(layer.Qidxs2, "_unsplit_weight")
            ) else None
            cb2 = cb if inv_rs != 0.0 else None
            out_sz = sum(layer.glq_shard_sizes)
            in_sz = layer.glq_in_features
            weight = _dequantize_glq_weight(
                q, su, sv, ws, cb,
                Qidxs2=q2, inv_resid_scale=inv_rs, codebook2=cb2,
                out_features=out_sz, in_features=in_sz,
            )
            layer.weight = torch.nn.Parameter(
                weight.to(device), requires_grad=False,
            )
            layer._glq_use_dense = True
            for attr in ["Qidxs", "SU", "SV", "Wscale", "Qidxs2", "inv_resid_scale"]:
                if hasattr(layer, attr):
                    delattr(layer, attr)
            return

        # Determine max_bpw across all shards
        max_bpw = 2
        if getattr(layer, "glq_is_fused", False):
            for i in range(layer.glq_num_shards):
                inv_rs = layer.inv_resid_scale.get_shard(i).item()
                if inv_rs != 0.0:
                    max_bpw = max(max_bpw, bpw)
        else:
            inv_rs = layer.inv_resid_scale.item()
            if inv_rs != 0.0:
                max_bpw = bpw

        _ensure_codebook(device, max_bpw=max_bpw)
        _try_load_cuda_ext()

        # Ensure all weight tensors on GPU
        for attr in ["Qidxs", "SU", "SV", "Wscale", "Qidxs2", "inv_resid_scale"]:
            t = getattr(layer, attr, None)
            if t is not None and hasattr(t, "device") and t.device != device:
                setattr(
                    layer, attr,
                    torch.nn.Parameter(t.data.to(device), requires_grad=False),
                )

        # Cache per-shard metadata
        if getattr(layer, "glq_is_fused", False):
            layer._glq_shard_meta = []
            n_pad = layer.glq_n_pad
            for i in range(layer.glq_num_shards):
                out_sz = layer.glq_shard_sizes[i]
                m_pad = _glq_pad(out_sz)
                inv_rs_val = layer.inv_resid_scale.get_shard(i).item()
                layer._glq_shard_meta.append({
                    "wscale": layer.Wscale.get_shard(i).item(),
                    "has_stage2": inv_rs_val != 0.0,
                    "inv_rs": inv_rs_val,
                    "out": out_sz,
                    "in": layer.glq_in_features,
                    "m_pad": m_pad,
                    "n_pad": n_pad,
                    "log_n": int(math.log2(n_pad)),
                    "log_m": int(math.log2(m_pad)),
                })
        else:
            inv_rs = layer.inv_resid_scale.item()
            layer._glq_wscale = layer.Wscale.item()
            layer._glq_has_stage2 = inv_rs != 0.0
            layer._glq_inv_rs = inv_rs
            if hasattr(layer, "Qidxs") and layer.Qidxs.dim() == 2:
                layer._glq_m_pad = layer.Qidxs.shape[0]
                layer._glq_n_pad = layer.Qidxs.shape[1] * 8
                layer._glq_out = layer.glq_out_features
                layer._glq_in = layer.glq_in_features
            else:
                layer._glq_m_pad = layer.glq_m_pad
                layer._glq_n_pad = layer.glq_n_pad
                layer._glq_out = layer.glq_out_features
                layer._glq_in = layer.glq_in_features
            layer._glq_log_n = int(math.log2(layer._glq_n_pad))
            layer._glq_log_m = int(math.log2(layer._glq_m_pad))

        # Remove weight_loader function refs (function pointers are not
        # picklable and break sglang's weight reloading pathways)
        for name, param in layer.named_parameters():
            try:
                if hasattr(param, "weight_loader") and not isinstance(
                    type(param).__dict__.get("weight_loader"), property
                ):
                    del param.weight_loader
            except (AttributeError, TypeError):
                pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Dequantized Mamba layers — standard dense matmul
        if getattr(layer, "_glq_use_dense", False):
            return F.linear(x, layer.weight.to(x.dtype), bias)

        orig_shape = x.shape
        in_features = layer.glq_in_features
        x = x.reshape(-1, in_features)
        device = x.device
        cb, cb2 = _codebook, _codebook2_small

        if getattr(layer, "glq_is_fused", False):
            shard_outputs = []
            for i in range(layer.glq_num_shards):
                meta = layer._glq_shard_meta[i]
                y_shard = _glq_apply_shard(
                    x, device, cb, cb2,
                    Qidxs=layer.Qidxs.get_shard(i),
                    SU=layer.SU.get_shard(i),
                    SV=layer.SV.get_shard(i),
                    wscale=meta["wscale"],
                    has_stage2=meta["has_stage2"],
                    inv_rs=meta["inv_rs"],
                    Qidxs2=layer.Qidxs2.get_shard(i) if meta["has_stage2"] else None,
                    out_features=meta["out"],
                    in_features=meta["in"],
                    m_pad=meta["m_pad"],
                    n_pad=meta["n_pad"],
                    log_n=meta["log_n"],
                    log_m=meta["log_m"],
                )
                shard_outputs.append(y_shard)
            y = torch.cat(shard_outputs, dim=-1)
            out_features = sum(layer.glq_shard_sizes)
        else:
            y = _glq_apply_single(x, layer, "", cb, cb2, device)
            out_features = layer.glq_out_features

        if bias is not None:
            y = y + bias.unsqueeze(0).to(y.dtype)

        return y.reshape(*orig_shape[:-1], out_features)
