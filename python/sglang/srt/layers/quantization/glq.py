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
from glq.hadamard import _block_decompose
from glq.quantized_linear import _pack_block_meta


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


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _detect_block_diag(m_pad: int, n_pad: int):
    """Return (is_block_diag, blocks_m, blocks_n) for a given buffer shape.

    Block-diagonal checkpoints (Phase B, post-v0.2.9) land non-power-of-2
    ``m_pad``/``n_pad`` equal to the true ``out_features``/``in_features``.
    Legacy pow2 checkpoints land the padded size. Decomposing the exact
    dim into a sum of power-of-2 blocks lets the fused kernel run the
    multiblock FHT without padding waste.
    """
    is_bd = not (_is_pow2(m_pad) and _is_pow2(n_pad))
    blocks_m = _block_decompose(m_pad) if is_bd else [m_pad]
    blocks_n = _block_decompose(n_pad) if is_bd else [n_pad]
    return is_bd, blocks_m, blocks_n


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
    """Register one set of GLQ compressed buffers on the layer.

    Includes stage-3 and stage-4 RVQ buffers so 5-8 bpw (Phase D) checkpoints
    load cleanly. Unused stages are zero-initialised and the forward path
    skips them based on the per-layer ``_glq_n_stages`` metadata.
    """
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
    # Phase D: N-stage RVQ for 5-8 bpw. Zero buffers are safe to ship —
    # forward() only reads them when _glq_n_stages >= 3 / >= 4.
    setattr(layer, f"Qidxs3{p}", _make_glq_param(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16)))
    setattr(layer, f"inv_resid_scale2{p}", _make_glq_param(
        torch.zeros((), dtype=torch.float32)))
    setattr(layer, f"Qidxs4{p}", _make_glq_param(
        torch.zeros(m_pad, n_blocks, dtype=torch.int16)))
    setattr(layer, f"inv_resid_scale3{p}", _make_glq_param(
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
    Qidxs3=None, inv_rs2=0.0, Qidxs4=None, inv_rs3=0.0,
    block_diag_meta=None,
):
    """Full GLQ forward for one set of weight buffers.

    Block-diagonal (Phase B) path: when ``block_diag_meta`` is provided, dispatches
    to the single-call ``glq_fused_linear_block_diag_cuda`` which handles
    input-RHT + dequant+matmul + output-RHT for non-power-of-2 dims in one
    host call (up to 3× fewer kernel launches, ~23× wall-time win on small
    models under lm-eval per Phase C).

    Pow2 path: legacy 3-call (input_rht + dequant_matmul + output_rht) kept
    for checkpoints quantized before block-diagonal FHT shipped.

    Stages 3-4 (Phase D, 5-8 bpw) reuse the primary 65536-entry E8 codebook —
    matches the convention in glq.quantized_linear's fallback path.
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
    if Qidxs3 is not None and Qidxs3.device != x.device:
        Qidxs3 = Qidxs3.to(device)
    if Qidxs4 is not None and Qidxs4.device != x.device:
        Qidxs4 = Qidxs4.to(device)

    primary_cb = cb.codebook_half
    cb2_half = cb2.codebook_half if has_stage2 and cb2 is not None else None

    # ── Block-diagonal fast path (Phase B + D) ──────────────────────
    if (block_diag_meta is not None and _use_cuda_c
            and hasattr(_ik._glq_cuda, "glq_fused_linear_block_diag_cuda")
            and n_pad <= 32768 and m_pad <= 32768):
        _empty_i16 = torch.empty(0, dtype=torch.int16, device=device)
        _empty_f16 = torch.empty(0, dtype=torch.float16, device=device)
        bn_tensor = block_diag_meta["blocks_n_tensor"]  # CPU int64
        bm_tensor = block_diag_meta["blocks_m_tensor"]
        # Lazy push of packed metadata to GPU (cache on the meta dict so
        # repeated forwards reuse it).
        bn_meta = block_diag_meta.get("blocks_n_meta_gpu")
        bm_meta = block_diag_meta.get("blocks_m_meta_gpu")
        if bn_meta is None or bn_meta.device != device:
            bn_meta = block_diag_meta["blocks_n_meta_cpu"].to(device, non_blocking=True)
            bm_meta = block_diag_meta["blocks_m_meta_cpu"].to(device, non_blocking=True)
            block_diag_meta["blocks_n_meta_gpu"] = bn_meta
            block_diag_meta["blocks_m_meta_gpu"] = bm_meta

        q2 = Qidxs2 if has_stage2 else _empty_i16
        cb2_arg = cb2_half if has_stage2 and cb2_half is not None else _empty_f16
        q3 = Qidxs3 if Qidxs3 is not None else _empty_i16
        cb3_arg = primary_cb if Qidxs3 is not None else _empty_f16
        q4 = Qidxs4 if Qidxs4 is not None else _empty_i16
        cb4_arg = primary_cb if Qidxs4 is not None else _empty_f16
        y = _ik._glq_cuda.glq_fused_linear_block_diag_cuda(
            x.half().contiguous(), SV, SU,
            Qidxs, primary_cb,
            float(wscale),
            in_features, out_features,
            n_pad, m_pad,
            bn_tensor, bm_tensor,
            bn_meta, bm_meta,
            q2, cb2_arg, float(inv_rs) if has_stage2 else 0.0,
            q3, cb3_arg, float(inv_rs2) if Qidxs3 is not None else 0.0,
            q4, cb4_arg, float(inv_rs3) if Qidxs4 is not None else 0.0,
        )
        if dtype != torch.float16:
            y = y.to(dtype)
        return y

    # ── Pow2 / legacy 3-call path ───────────────────────────────────

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
    cb3_arg = primary_cb if Qidxs3 is not None else None
    cb4_arg = primary_cb if Qidxs4 is not None else None

    y_rht = glq_dequant_matmul(
        x_rht, Qidxs, primary_cb, wscale,
        Qidxs2=Qidxs2, codebook2=cb2_half,
        inv_resid_scale=inv_rs, codebook_packed=cb_packed,
        Qidxs3=Qidxs3, codebook3=cb3_arg, inv_resid_scale2=inv_rs2,
        Qidxs4=Qidxs4, codebook4=cb4_arg, inv_resid_scale3=inv_rs3,
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
    # Phase D N-stage: stage 3/4 tensors when _glq_n_stages >= 3 / 4.
    n_stages = getattr(layer, f"_glq_n_stages{prefix}", 2 if has_stage2 else 1)
    inv_rs2 = getattr(layer, f"_glq_inv_rs2{prefix}", 0.0)
    inv_rs3 = getattr(layer, f"_glq_inv_rs3{prefix}", 0.0)
    Qidxs3 = (getattr(layer, f"Qidxs3{prefix}").to(device)
              if n_stages >= 3 else None)
    Qidxs4 = (getattr(layer, f"Qidxs4{prefix}").to(device)
              if n_stages >= 4 else None)
    # Phase B block-diagonal metadata — set by process_weights_after_loading.
    block_diag_meta = getattr(layer, f"_glq_bd_meta{prefix}", None)
    return _glq_apply_shard(
        x, device, cb, cb2,
        Qidxs=Qidxs, SU=SU, SV=SV, wscale=wscale,
        has_stage2=has_stage2, inv_rs=inv_rs, Qidxs2=Qidxs2,
        out_features=out_features, in_features=in_features,
        m_pad=m_pad, n_pad=n_pad, log_n=log_n, log_m=log_m,
        Qidxs3=Qidxs3, inv_rs2=inv_rs2,
        Qidxs4=Qidxs4, inv_rs3=inv_rs3,
        block_diag_meta=block_diag_meta,
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
            # Phase D: N-stage RVQ — stage 3 (5/6bpw) and stage 4 (7/8bpw).
            # Zero-initialised; forward skips them when _glq_n_stages < 3 / 4.
            layer.Qidxs3 = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16,
                weight_loader=weight_loader,
            )
            layer.inv_resid_scale2 = GLQShardedParameter(
                [1] * len(output_partition_sizes), 0, torch.float32,
                weight_loader=weight_loader,
            )
            layer.Qidxs4 = GLQShardedParameter(
                output_partition_sizes, n_blocks, torch.int16,
                weight_loader=weight_loader,
            )
            layer.inv_resid_scale3 = GLQShardedParameter(
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

        # Ensure all weight tensors on GPU (includes Phase D stage 3/4).
        for attr in ["Qidxs", "SU", "SV", "Wscale", "Qidxs2", "inv_resid_scale",
                     "Qidxs3", "inv_resid_scale2", "Qidxs4", "inv_resid_scale3"]:
            t = getattr(layer, attr, None)
            if t is not None and hasattr(t, "device") and t.device != device:
                setattr(
                    layer, attr,
                    torch.nn.Parameter(t.data.to(device), requires_grad=False),
                )

        # Cache per-shard metadata
        if getattr(layer, "glq_is_fused", False):
            layer._glq_shard_meta = []
            # Phase B: recover actual (possibly non-pow2) shard shapes from the
            # loaded per-shard Qidxs buffers. At register time we sized to
            # pow2; the weight loader overwrote with whatever the checkpoint
            # contained. A 576-wide shard here means block-diagonal.
            for i in range(layer.glq_num_shards):
                out_sz = layer.glq_shard_sizes[i]
                qidxs_i = layer.Qidxs.get_shard(i)
                if qidxs_i.dim() == 2:
                    m_pad = qidxs_i.shape[0]
                    n_pad = qidxs_i.shape[1] * 8
                else:
                    m_pad = _glq_pad(out_sz)
                    n_pad = layer.glq_n_pad
                inv_rs_val = layer.inv_resid_scale.get_shard(i).item()
                # Phase D: detect active stage count from non-zero inv_resid_scale*.
                inv_rs2_val = (layer.inv_resid_scale2.get_shard(i).item()
                               if hasattr(layer, "inv_resid_scale2") else 0.0)
                inv_rs3_val = (layer.inv_resid_scale3.get_shard(i).item()
                               if hasattr(layer, "inv_resid_scale3") else 0.0)
                if inv_rs3_val != 0.0:
                    n_stages = 4
                elif inv_rs2_val != 0.0:
                    n_stages = 3
                elif inv_rs_val != 0.0:
                    n_stages = 2
                else:
                    n_stages = 1
                is_bd, blocks_m, blocks_n = _detect_block_diag(m_pad, n_pad)
                bd_meta = None
                if is_bd:
                    bd_meta = {
                        "blocks_n_tensor": torch.tensor(
                            blocks_n, dtype=torch.int64, device="cpu"),
                        "blocks_m_tensor": torch.tensor(
                            blocks_m, dtype=torch.int64, device="cpu"),
                        "blocks_n_meta_cpu": _pack_block_meta(blocks_n),
                        "blocks_m_meta_cpu": _pack_block_meta(blocks_m),
                        "blocks_n_meta_gpu": None,
                        "blocks_m_meta_gpu": None,
                    }
                layer._glq_shard_meta.append({
                    "wscale": layer.Wscale.get_shard(i).item(),
                    "has_stage2": inv_rs_val != 0.0,
                    "inv_rs": inv_rs_val,
                    "n_stages": n_stages,
                    "inv_rs2": inv_rs2_val,
                    "inv_rs3": inv_rs3_val,
                    "out": out_sz,
                    "in": layer.glq_in_features,
                    "m_pad": m_pad,
                    "n_pad": n_pad,
                    "log_n": int(math.log2(n_pad)) if _is_pow2(n_pad) else 0,
                    "log_m": int(math.log2(m_pad)) if _is_pow2(m_pad) else 0,
                    "bd_meta": bd_meta,
                })
        else:
            inv_rs = layer.inv_resid_scale.item()
            layer._glq_wscale = layer.Wscale.item()
            layer._glq_has_stage2 = inv_rs != 0.0
            layer._glq_inv_rs = inv_rs
            # Phase D: detect N-stage from non-zero inv_resid_scale*.
            inv_rs2 = (layer.inv_resid_scale2.item()
                       if hasattr(layer, "inv_resid_scale2") else 0.0)
            inv_rs3 = (layer.inv_resid_scale3.item()
                       if hasattr(layer, "inv_resid_scale3") else 0.0)
            if inv_rs3 != 0.0:
                layer._glq_n_stages = 4
            elif inv_rs2 != 0.0:
                layer._glq_n_stages = 3
            elif inv_rs != 0.0:
                layer._glq_n_stages = 2
            else:
                layer._glq_n_stages = 1
            layer._glq_inv_rs2 = inv_rs2
            layer._glq_inv_rs3 = inv_rs3
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
            # Phase B: detect block-diag from the loaded (possibly non-pow2)
            # buffer shape, build block decomposition + packed metadata so
            # the forward path can dispatch to glq_fused_linear_block_diag_cuda.
            is_bd, blocks_m, blocks_n = _detect_block_diag(
                layer._glq_m_pad, layer._glq_n_pad)
            if is_bd:
                layer._glq_bd_meta = {
                    "blocks_n_tensor": torch.tensor(
                        blocks_n, dtype=torch.int64, device="cpu"),
                    "blocks_m_tensor": torch.tensor(
                        blocks_m, dtype=torch.int64, device="cpu"),
                    "blocks_n_meta_cpu": _pack_block_meta(blocks_n),
                    "blocks_m_meta_cpu": _pack_block_meta(blocks_m),
                    "blocks_n_meta_gpu": None,
                    "blocks_m_meta_gpu": None,
                }
                # Log-of-zero is harmless for the block-diag path (kernel
                # reads per-block log2 from the metadata) but keep legacy
                # log values valid for the fallback path in case it's hit.
                layer._glq_log_n = (int(math.log2(layer._glq_n_pad))
                                     if _is_pow2(layer._glq_n_pad) else 0)
                layer._glq_log_m = (int(math.log2(layer._glq_m_pad))
                                     if _is_pow2(layer._glq_m_pad) else 0)
            else:
                layer._glq_bd_meta = None
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
        orig_shape = x.shape
        in_features = layer.glq_in_features
        x = x.reshape(-1, in_features)
        device = x.device
        cb, cb2 = _codebook, _codebook2_small

        if getattr(layer, "glq_is_fused", False):
            shard_outputs = []
            for i in range(layer.glq_num_shards):
                meta = layer._glq_shard_meta[i]
                n_stages = meta.get("n_stages", 2 if meta["has_stage2"] else 1)
                Qidxs3 = (layer.Qidxs3.get_shard(i)
                          if n_stages >= 3 and hasattr(layer, "Qidxs3") else None)
                Qidxs4 = (layer.Qidxs4.get_shard(i)
                          if n_stages >= 4 and hasattr(layer, "Qidxs4") else None)
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
                    Qidxs3=Qidxs3, inv_rs2=meta.get("inv_rs2", 0.0),
                    Qidxs4=Qidxs4, inv_rs3=meta.get("inv_rs3", 0.0),
                    block_diag_meta=meta.get("bd_meta"),
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
