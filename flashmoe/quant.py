"""Block-scaled quantization of expert weights for flashmoe.QuantType.MXFP8, MXFP4 and NVFP4.

MXFP8: E4M3 elements and one UE8M0 scale (a power of two) per 32 consecutive elements along K, following the
NVIDIA MXFP8 recipe scale = 2^ceil(log2(amax / 448)). Scale factors use the 128x4 swizzled layout of
cuBLASLt (CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) and CUTLASS (Sm1xxBlockScaledConfig<32>), the same as
csrc/include/flashmoe/infra/mx.cuh, which the kernel uses to quantize activations.

MXFP4: E2M1 elements (two per byte, the even K element in the low nibble) and one UE8M0 scale per 32 consecutive
elements along K, scale = 2^ceil(log2(amax / 6)) so that a block never saturates. Same scale factor layout as MXFP8.

NVFP4: E2M1 elements and one UE4M3 scale per 16 consecutive elements along K (CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
Sm1xxBlockScaledConfig<16>), plus one FP32 global decode scale per tensor, gs = amax / (6 * 448):
    x ~ e2m1 * float(sf) * gs.
The block scale is e4m3(amax_block / 6 / gs), see quantize_nvfp4.

E2M1 rounds to nearest, ties to the even code, and saturates at 6 (cvt.rn.satfinite.e2m1x2.f32).
"""
import torch

BLOCK_SIZE = 32  # K elements per scale factor of MXFP8 and MXFP4
NVFP4_BLOCK_SIZE = 16  # K elements per scale factor of NVFP4
E4M3_MAX = 448.0
E2M1_MAX = 6.0
_UE8M0_MIN = 1
_UE8M0_NAN = 255
_FLT_MAX = torch.finfo(torch.float32).max
_FLT_MIN = torch.finfo(torch.float32).tiny
# value of E2M1 code 0..7, the sign is bit 3: 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def sf_bytes(rows: int, k: int, block_size: int = BLOCK_SIZE) -> int:
    """Bytes of the scale factors of a [rows, k] matrix: rows padded to 128, k / block_size blocks padded to 4.
    block_size is 32 for MXFP8 and MXFP4 and 16 for NVFP4."""
    assert k % block_size == 0
    k_blocks = k // block_size
    return ((rows + 127) // 128 * 128) * ((k_blocks + 3) // 4 * 4)


def _sf_offsets(rows: int, k_blocks: int, device) -> torch.Tensor:
    r = torch.arange(rows, device=device, dtype=torch.int64).view(-1, 1)
    kb = torch.arange(k_blocks, device=device, dtype=torch.int64).view(1, -1)
    k_tiles = (k_blocks + 3) // 4
    return ((r // 128) * k_tiles + kb // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + kb % 4


def _ue8m0_from_amax(amax: torch.Tensor, shift: int = 8, threshold: int = 0x600000) -> torch.Tensor:
    """Biased exponent of the smallest power of two s with amax / s <= the largest element value (bit-exact with
    mx.cuh). The defaults are for E4M3: 448 = 1.75 * 2^8, hence e = biased(amax) - 8 + (mantissa(amax) > 0.75).
    E2M1 (6 = 1.5 * 2^2) uses shift = 2 and threshold = 0x400000 (mantissa > 0.5)."""
    bits = amax.float().abs().view(torch.int32)
    biased = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    e = (biased - shift + (mantissa > threshold).to(torch.int32)).clamp(min=_UE8M0_MIN)
    return torch.where(biased == 0xFF, torch.full_like(e, _UE8M0_NAN), e).to(torch.uint8)


def _scatter_sf(sf_values: torch.Tensor, n: int, k_blocks: int, device) -> torch.Tensor:
    """Writes the [B, n, k_blocks] uint8 scales into B canonical (128x4 swizzled) scale factor matrices [B, bytes]."""
    size = (n + 127) // 128 * 128 * ((k_blocks + 3) // 4 * 4)
    out = torch.zeros(sf_values.shape[0], size, dtype=torch.uint8, device=device)
    out[:, _sf_offsets(n, k_blocks, device).flatten()] = sf_values.reshape(sf_values.shape[0], -1)
    return out


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


# ----------------------------------------------------------------------------------------------------------------
# FP4: E2M1 elements, MXFP4 and NVFP4
# ----------------------------------------------------------------------------------------------------------------

def _e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """float32 -> E2M1 codes (uint8, 0..15, the sign is bit 3): round to nearest, ties to the even code, saturating at 6.
    Bit-identical to cvt.rn.satfinite.e2m1x2.f32: the sign of -0 and of values that round to zero is kept, NaN maps to
    +6 (the quantizers carry NaN in the block scale instead).

    The midpoints of the grid {0, .5, 1, 1.5, 2, 3, 4, 6} are 0.25, 0.75, 1.25, 1.75, 2.5, 3.5 and 5; a tie goes to the
    neighbor with the even code, i.e., 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4."""
    a = x.abs()
    code = ((a > 0.25).to(torch.int32) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0))
    code = code | (torch.signbit(x).to(torch.int32) << 3)
    return torch.where(torch.isnan(x), torch.full_like(code, 7), code).to(torch.uint8)


def _e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (uint8) -> float32"""
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=code.device)
    v = values[(code & 7).long()]
    return torch.where((code & 8) != 0, -v, v)


def _pack_e2m1(code: torch.Tensor) -> torch.Tensor:
    """[..., K] codes -> [..., K / 2] bytes, the even K element in the low nibble"""
    return code[..., 0::2] | (code[..., 1::2] << 4)


def _unpack_e2m1(q: torch.Tensor) -> torch.Tensor:
    """[..., K / 2] bytes -> [..., K] codes"""
    return torch.stack((q & 0xF, q >> 4), dim=-1).reshape(*q.shape[:-1], 2 * q.shape[-1])


def nvfp4_scale(amax) -> torch.Tensor:
    """Global decode scale of an NVFP4 tensor with absolute maximum amax (float or tensor): amax / (6 * 448), so that
    the largest block scale is the largest finite E4M3 value. 1 where amax == 0 (an all-zero tensor), and never below
    the smallest normal FP32 value, so that its reciprocal is finite."""
    amax = torch.as_tensor(amax, dtype=torch.float32)
    gs = (amax / (E2M1_MAX * E4M3_MAX)).clamp(min=_FLT_MIN)
    return torch.where(amax > 0, gs, torch.ones_like(gs))


def quantize_mxfp4(w: torch.Tensor):
    """Quantizes K-major weights [..., N, K] (e.g. [num_local_experts, I, H] for the up projections and
    [num_local_experts, H, I] for the down projection) to MXFP4.

    Returns (q, sf): q is [..., N, K / 2] torch.uint8, two E2M1 elements per byte, the even K element in the low
    nibble, and sf is [..., sf_bytes(N, K)] torch.uint8, one contiguous scale factor matrix per leading index (one
    UE8M0 per 32 K elements in the canonical layout), as expected by flashmoe.ForwardArgs(local_expert_*_sf=...).
    The scale of a block is 2^ceil(log2(amax / 6)), element = round_e2m1(x / scale).
    """
    *lead, n, k = w.shape
    assert k % BLOCK_SIZE == 0, "K must be a multiple of 32"
    x = w.float().reshape(-1, n, k // BLOCK_SIZE, BLOCK_SIZE)
    # E2M1: 6 = 1.5 * 2^2
    e = _ue8m0_from_amax(x.abs().amax(dim=-1), shift=2, threshold=0x400000)  # [B, N, K / 32]
    rcp = torch.ldexp(torch.ones_like(x[..., 0]), 127 - e.to(torch.int32))  # 2^-(e - 127), exact in FP32
    code = _e2m1_encode(x * rcp.unsqueeze(-1)).reshape(-1, n, k)
    q = _pack_e2m1(code).reshape(*lead, n, k // 2)
    return q, _scatter_sf(e, n, k // BLOCK_SIZE, w.device).reshape(*lead, -1)


def quantize_nvfp4(w: torch.Tensor, gs: torch.Tensor = None):
    """Quantizes K-major weights [..., N, K] to NVFP4 (the same shapes as quantize_mxfp4).

    Returns (q, sf, gs): q is [..., N, K / 2] torch.uint8 (two E2M1 per byte, the even K element in the low nibble),
    sf is [..., sf_bytes(N, K, 16)] torch.uint8 (the bytes of one torch.float8_e4m3fn scale per 16 K elements in the
    canonical layout, one matrix per leading index) and gs is the float32 global decode scale [...] of each matrix
    along the leading dimensions, a dequantized element is e2m1 * float(sf) * gs.
    gs = nvfp4_scale(amax of the matrix) unless given (a scalar or a tensor broadcastable to [...]), e.g., a
    static per-layer scale. With st = 1 / gs: sf = e4m3(amax_block * (st / 6)), enc = min(st / float(sf), FLT_MAX),
    element = round_e2m1(x * enc), the recipe of CUTLASS compute_1d_scaling_factor_and_quantized_output.
    """
    *lead, n, k = w.shape
    assert k % NVFP4_BLOCK_SIZE == 0, "K must be a multiple of 16"
    x = w.float().reshape(-1, n, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    if gs is None:
        gs = nvfp4_scale(x.abs().amax(dim=(1, 2, 3)).reshape(lead))
    gs = torch.as_tensor(gs, dtype=torch.float32, device=w.device).expand(lead).reshape(-1)  # [B]
    st = (1.0 / gs).view(-1, 1, 1)  # encode scale, IEEE division
    amax = x.abs().amax(dim=-1)  # [B, N, K / 16]
    sf = (amax * (st * (1.0 / 6.0))).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)  # round to nearest even
    enc = (st / sf.float()).clamp(max=_FLT_MAX)  # sf == 0 gives FLT_MAX: the block dequantizes to 0 either way
    code = _e2m1_encode(x * enc.unsqueeze(-1)).reshape(-1, n, k)
    q = _pack_e2m1(code).reshape(*lead, n, k // 2)
    sf_out = _scatter_sf(sf.view(torch.uint8), n, k // NVFP4_BLOCK_SIZE, w.device).reshape(*lead, -1)
    return q, sf_out, gs.reshape(lead)


def dequantize_fp4(q: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor = None, fmt: str = None,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of quantize_mxfp4 / quantize_nvfp4: q [..., N, K / 2] uint8 and sf [..., sf bytes] uint8 give
    [..., N, K]. fmt is "mxfp4" or "nvfp4" (default: "nvfp4" if gs is given, else "mxfp4"); gs [...] is the global
    decode scale of NVFP4 (None means 1)."""
    if fmt is None:
        fmt = "mxfp4" if gs is None else "nvfp4"
    assert fmt in ("mxfp4", "nvfp4"), "fmt should be mxfp4 or nvfp4"
    *lead, n, kh = q.shape
    k = kh * 2
    block = BLOCK_SIZE if fmt == "mxfp4" else NVFP4_BLOCK_SIZE
    k_blocks = k // block
    s = sf.reshape(-1, sf.shape[-1])[:, _sf_offsets(n, k_blocks, q.device).flatten()]  # [B, N * K / block]
    if fmt == "mxfp4":
        scale = torch.ldexp(torch.ones(s.shape, device=q.device), s.to(torch.int32) - 127)
        scale = torch.where(s == _UE8M0_NAN, torch.full_like(scale, float("nan")), scale)
    else:
        scale = s.contiguous().view(torch.float8_e4m3fn).float()
    x = _e2m1_decode(_unpack_e2m1(q)).reshape(-1, n, k_blocks, block) * scale.reshape(-1, n, k_blocks, 1)
    if fmt == "nvfp4" and gs is not None:
        gs = torch.as_tensor(gs, dtype=torch.float32, device=q.device).expand(lead).reshape(-1, 1, 1, 1)
        x = x * gs
    return x.reshape(*lead, n, k).to(dtype)
