"""MXFP8 quantization of expert weights for flashmoe.QuantType.MXFP8.

MXFP8: E4M3 elements and one UE8M0 scale (a power of two) per 32 consecutive elements along K, following the
NVIDIA MXFP8 recipe scale = 2^ceil(log2(amax / 448)). Scale factors use the 128x4 swizzled layout of
cuBLASLt (CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) and CUTLASS (Sm1xxBlockScaledConfig<32>), the same as
csrc/include/flashmoe/infra/mx.cuh, which the kernel uses to quantize activations.
"""
import torch

BLOCK_SIZE = 32
E4M3_MAX = 448.0
_UE8M0_MIN = 1
_UE8M0_NAN = 255


def sf_bytes(rows: int, k: int) -> int:
    """Bytes of the scale factors of a [rows, k] matrix: rows padded to 128, k / 32 blocks padded to 4."""
    assert k % BLOCK_SIZE == 0
    k_blocks = k // BLOCK_SIZE
    return ((rows + 127) // 128 * 128) * ((k_blocks + 3) // 4 * 4)


def _sf_offsets(rows: int, k_blocks: int, device) -> torch.Tensor:
    r = torch.arange(rows, device=device, dtype=torch.int64).view(-1, 1)
    kb = torch.arange(k_blocks, device=device, dtype=torch.int64).view(1, -1)
    k_tiles = (k_blocks + 3) // 4
    return ((r // 128) * k_tiles + kb // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + kb % 4


def _ue8m0_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Biased exponent of the smallest power of two s with amax / s <= 448 (bit-exact with mx.cuh)."""
    bits = amax.float().abs().view(torch.int32)
    biased = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    e = (biased - 8 + (mantissa > 0x600000).to(torch.int32)).clamp(min=_UE8M0_MIN)
    return torch.where(biased == 0xFF, torch.full_like(e, _UE8M0_NAN), e).to(torch.uint8)


def quantize_mxfp8(w: torch.Tensor):
    """Quantizes K-major weights [..., N, K] (e.g. [num_local_experts, I, H] for the up projections and
    [num_local_experts, H, I] for the down projection) to MXFP8.

    Returns (q, sf): q is [..., N, K] torch.float8_e4m3fn and sf is [..., sf_bytes(N, K)] torch.uint8, one
    contiguous scale factor matrix per leading index, as expected by flashmoe.ForwardArgs(local_expert_*_sf=...).
    """
    *lead, n, k = w.shape
    assert k % BLOCK_SIZE == 0, "K must be a multiple of 32"
    x = w.float().reshape(-1, n, k // BLOCK_SIZE, BLOCK_SIZE)
    e = _ue8m0_from_amax(x.abs().amax(dim=-1))  # [B, N, K / 32]
    # x / 2^(e - 127), exact in FP32, then round to nearest even E4M3 (|x| <= 448 by construction)
    rcp = torch.ldexp(torch.ones_like(x[..., 0]), 127 - e.to(torch.int32))
    q = (x * rcp.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(*lead, n, k)
    sf = torch.zeros(x.shape[0], sf_bytes(n, k), dtype=torch.uint8, device=w.device)
    sf[:, _sf_offsets(n, k // BLOCK_SIZE, w.device).flatten()] = e.reshape(x.shape[0], -1)
    return q, sf.reshape(*lead, -1)


def dequantize_mxfp8(q: torch.Tensor, sf: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of quantize_mxfp8."""
    *lead, n, k = q.shape
    k_blocks = k // BLOCK_SIZE
    e = sf.reshape(-1, sf.shape[-1])[:, _sf_offsets(n, k_blocks, q.device).flatten()]
    scale = torch.ldexp(torch.ones(e.shape, device=q.device), e.to(torch.int32) - 127)
    scale = torch.where(e == _UE8M0_NAN, torch.full_like(scale, float("nan")), scale)
    x = q.float().reshape(-1, n, k_blocks, BLOCK_SIZE) * scale.reshape(-1, n, k_blocks, 1)
    return x.reshape(*lead, n, k).to(dtype)


def blockfp8_to_mxfp8(w: torch.Tensor, scale_inv: torch.Tensor, block_size=(128, 128)):
    """Converts DeepSeek-style block FP8 weights (e4m3 [..., N, K] with float32 scale_inv [..., N / 128, K / 128])
    to MXFP8 by dequantizing and re-quantizing with 1x32 power-of-two scales."""
    bn, bk = block_size
    *lead, n, k = w.shape
    s = scale_inv.float().repeat_interleave(bn, dim=-2)[..., :n, :].repeat_interleave(bk, dim=-1)[..., :k]
    return quantize_mxfp8(w.float() * s)
