"""Measure empirical CUDA ceilings and plot saved kernel timing records."""
import argparse
import json
import statistics
from pathlib import Path

import torch


def measure(call):
    """Return median device time in seconds after warmup."""
    for _ in range(5):
        call()
    samples = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1000)
    return statistics.median(samples)


def main():
    """Plot one GPU and dtype with measured copy and matrix multiply ceilings."""
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Roofline calibration requires CUDA")
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not records:
        parser.error("Expected at least one timing record")
    hardware = torch.cuda.get_device_name()
    dtype_name = records[0]["dtype"]
    for record in records:
        if (record["device"], record["hardware"], record["dtype"]) != ("cuda", hardware, dtype_name):
            parser.error("All inputs must match the calibration device and dtype")
    dtype = getattr(torch, dtype_name)
    # A 256 MiB source buffer exceeds L2 on the initial calibration targets.
    source = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda").fill_(17)
    destination = torch.empty_like(source)
    bandwidth = 2 * source.numel() / measure(lambda: destination.copy_(source))
    del source, destination
    size = 4096
    left, right = [torch.randn(size, size, dtype=dtype, device="cuda") for _ in range(2)]
    product = torch.empty_like(left)
    compute = 2 * size**3 / measure(lambda: torch.mm(left, right, out=product))
    intensities = [item["modeled_flops"] / item["minimum_bytes"] for item in records]
    ridge = compute / bandwidth
    bounds = [min(min(intensities), ridge)/2, ridge, max(max(intensities), ridge)*2]
    fig, ax = plt.subplots()
    ax.loglog(bounds, [min(x*bandwidth, compute)/1e9 for x in bounds], label="Measured roof")
    for item, intensity in zip(records, intensities):
        ax.scatter(intensity, item["modeled_gflops_s"], label=f'{item["backend"]} {item["kernel"]} N={item["tokens"]}')
    ax.set_xlabel("Modeled FLOP / minimum byte")
    ax.set_ylabel("Modeled GFLOP/s")
    ax.set_title(f"{hardware} · {dtype_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output)
    print(json.dumps(dict(device=hardware, dtype=dtype_name, copy_bytes_per_sec=bandwidth,
                          gemm_flops_per_sec=compute, plot=str(args.output))))


if __name__ == "__main__":
    main()
