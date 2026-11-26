"""CUDA forward kernels for contiguous FP16 and BF16 inputs."""
import torch
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({"M": 32, "K": 32}), triton.Config({"M": 64, "K": 32})], key=["N", "D"])
@triton.jit
def _attention(Q, KEY, V, OUT, N: tl.constexpr, D: tl.constexpr, M: tl.constexpr, K: tl.constexpr):
    """Accumulate causal attention with a running softmax maximum and denominator."""
    start = tl.program_id(0) * M
    base = tl.program_id(1).to(tl.int64) * N * D
    row = start + tl.arange(0, M)
    depth = tl.arange(0, D)
    q = tl.load(Q + base + row[:, None] * D + depth[None, :], mask=row[:, None] < N, other=0)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.full((M,), 0, tl.float32)
    acc = tl.full((M, D), 0, tl.float32)
    for tile in range(0, tl.cdiv(start + M, K)):
        col = tile * K + tl.arange(0, K)
        index = base + col[:, None] * D + depth[None, :]
        k = tl.load(KEY + index, mask=col[:, None] < N, other=0)
        v = tl.load(V + index, mask=col[:, None] < N, other=0)
        scores = tl.dot(q, tl.trans(k)) * (D ** -0.5)
        scores = tl.where((row[:, None] >= col[None, :]) & (col[None, :] < N), scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        scale = tl.exp(maximum - next_max)
        p = tl.exp(scores - next_max[:, None])
        denominator = denominator * scale + tl.sum(p, 1)
        acc = acc * scale[:, None] + tl.dot(p.to(v.dtype), v)
        maximum = next_max
    tl.store(OUT + base + row[:, None] * D + depth[None, :], acc / denominator[:, None], mask=row[:, None] < N)


@triton.jit
def _rms(X, R, W, OUT, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    """Normalize one row per program and mask its tail."""
    col = tl.arange(0, BLOCK)
    index = tl.program_id(0).to(tl.int64) * D + col
    x = tl.load(X + index, mask=col < D, other=0).to(tl.float32)
    r = tl.load(R + index, mask=col < D, other=0).to(tl.float32)
    w = tl.load(W + col, mask=col < D, other=0).to(tl.float32)
    total = x + r
    inv = tl.rsqrt(tl.sum(total * total, 0) / D + EPS)
    tl.store(OUT + index, total * inv * w, mask=col < D)


def attention(q, k, v):
    """Launch attention tiles on the input device."""
    batch, heads, n, d = q.shape
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _attention[lambda meta: (triton.cdiv(n, meta["M"]), batch * heads)](q, k, v, out, n, d)
    return out


def rms_norm(x, residual, weight, eps):
    """Launch the rowwise residual RMSNorm kernel."""
    d = x.size(-1)
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rms[(x.numel() // d,)](x, residual, weight, out, d, eps, triton.next_power_of_2(d))
    return out
