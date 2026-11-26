"""Time forward inference calls and report modeled traffic for roofline analysis."""
import argparse
import json
import platform
import statistics
import time

import torch

from . import causal_attention, residual_rms_norm


def main():
    """Write one JSON timing record per kernel with raw latency samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.tokens, args.depth, args.repeats) <= 0:
        parser.error("Dimensions and repeats must be positive")
    torch.manual_seed(0)
    n, d = args.tokens, args.depth
    options = {"device": args.device, "dtype": getattr(torch, args.dtype)}
    q, k, v = [torch.randn(1, 1, n, d, **options) for _ in range(3)]
    weight = torch.randn(d, **options)
    cases = [
        ("attention", lambda: causal_attention(q, k, v, backend=args.backend), 2*n*(n+1)*d, 4*n*d),
        ("rms_norm", lambda: residual_rms_norm(q, k, weight, backend=args.backend), 5*n*d, 3*n*d+d),
    ]
    for name, call, flops, elements in cases:
        for _ in range(5):
            call()
        samples = []
        for _ in range(args.repeats):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            call()
            if args.device == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter_ns() - start)
        median = statistics.median(samples)
        print(json.dumps(dict(kernel=name, **vars(args), torch_version=str(torch.__version__),
                              hardware=torch.cuda.get_device_name() if args.device == "cuda" else platform.machine(),
                              latency_ns=samples, median_ns=median, modeled_flops=flops,
                              minimum_bytes=elements*q.element_size(), modeled_gflops_s=flops/median)))


if __name__ == "__main__":
    main()
