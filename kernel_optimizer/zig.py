"""Call the native CPU kernels with borrowed PyTorch storage."""
import ctypes
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _library():
    """Locate the wheel library or a local Zig build."""
    directory = Path(__file__).resolve().parent
    paths = [directory / ("_cpu" + suffix) for suffix in EXTENSION_SUFFIXES]
    install = directory.parent / "zig-out"
    paths += [install / "lib" / name for name in ("libkernel_optimizer.so", "libkernel_optimizer.dylib")]
    paths += [install / "bin" / "kernel_optimizer.dll"]
    for path in paths:
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            pointer = ctypes.c_void_p
            size = ctypes.c_size_t
            lib.ko_attention.argtypes = [ctypes.c_uint32] + [pointer] * 5 + [size] * 3
            lib.ko_attention.restype = ctypes.c_int
            lib.ko_rms_norm.argtypes = [ctypes.c_uint32] + [pointer] * 4 + [size] * 2 + [ctypes.c_float]
            lib.ko_rms_norm.restype = ctypes.c_int
            return lib
    return None


def available():
    """Return whether this installation contains the native backend."""
    return _library() is not None


def _native():
    """Provide a build instruction when an explicit Zig request lacks a library."""
    lib = _library()
    if lib is None:
        raise RuntimeError("Install the project wheel or run zig build -Doptimize=ReleaseFast")
    return lib


_FORMAT = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def attention(q, k, v):
    """Normalize layout and pass the complete batch through the fused kernel."""
    q, k, v = (t.resolve_neg().contiguous() for t in (q, k, v))
    b, h, n, d = q.shape
    output = torch.empty_like(q)
    scratch = torch.empty(d, device=q.device, dtype=torch.float32)
    status = _native().ko_attention(_FORMAT[q.dtype], q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                    output.data_ptr(), scratch.data_ptr(), b * h, n, d)
    if status:
        raise ValueError("Native attention rejected the dimensions or dtype")
    return output


def rms_norm(x, residual, weight, eps):
    """Pass contiguous activations and weights through the native row kernel."""
    x, residual, weight = (t.resolve_neg().contiguous() for t in (x, residual, weight))
    output = torch.empty_like(x)
    depth = x.shape[-1]
    status = _native().ko_rms_norm(_FORMAT[x.dtype], x.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                                   output.data_ptr(), x.numel() // depth, depth, eps)
    if status:
        raise ValueError("Native RMSNorm requires a finite positive FP32 epsilon")
    return output
