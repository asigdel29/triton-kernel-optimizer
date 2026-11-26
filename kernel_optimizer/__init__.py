"""Inference APIs for reference and Triton implementations."""
import math

import torch
from torch.nn import functional as F


def _check_inputs(inputs, backend):
    """Check the common device, precision, and execution requirements."""
    first = inputs[0]
    if any(t.dtype != first.dtype or t.device != first.device for t in inputs):
        raise ValueError("Inputs must share a dtype and device")
    if first.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Expected FP16, BF16, or FP32")
    if backend not in ("torch", "triton"):
        raise ValueError("Expected backend torch or triton")
    if backend == "triton":
        if not first.is_cuda or first.dtype == torch.float32:
            raise ValueError("Triton requires CUDA and FP16 or BF16")
        if any(not t.is_contiguous() for t in inputs):
            raise ValueError("Triton requires contiguous inputs")


@torch.no_grad()
def causal_attention(query, key, value, *, backend="torch"):
    """Return causal self-attention for equal [batch, heads, tokens, depth] tensors."""
    _check_inputs((query, key, value), backend)
    if query.ndim != 4 or min(query.shape) < 1 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("Expected equal, nonempty [batch, heads, tokens, depth] inputs")
    if backend == "torch":
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)
    if query.size(-1) not in (16, 32, 64, 128):
        raise ValueError("Triton supports head dimensions 16, 32, 64, and 128")
    from .gpu import attention
    return attention(query, key, value)


@torch.no_grad()
def residual_rms_norm(x, residual, weight, *, eps=1e-6, backend="torch"):
    """Add the residual in FP32 and normalize the last axis with a learned weight."""
    _check_inputs((x, residual, weight), backend)
    if x.ndim < 1 or min(x.shape) < 1 or x.shape != residual.shape or weight.shape != (x.size(-1),):
        raise ValueError("Expected equal nonempty activations and one weight per channel")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected a finite positive epsilon")
    if backend == "torch":
        total = x.float() + residual.float()
        return (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
    if x.size(-1) > 16384:
        raise ValueError("Triton supports at most 16384 channels")
    from .gpu import rms_norm
    return rms_norm(x, residual, weight, eps)
