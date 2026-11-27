# Run and interpret a kernel experiment

## Check the reference on a CPU

1. Complete the Python setup in the [quickstart](../README.md#quickstart).
2. Run `make check`.
3. Run `make demo`.

The demo emits two JSON records. Each record names the backend, kernel, hardware, dtype, token count, and depth. A CPU-only run skips the Triton parity test.

```python
import torch
from kernel_optimizer import causal_attention, residual_rms_norm

q = torch.randn(1, 2, 17, 32)
output = causal_attention(q, q, q)
normalized = residual_rms_norm(output, q, torch.ones(32))
assert normalized.shape == q.shape
```

## Collect comparable trials

Activate the repository's Python environment. Use a fresh output directory for each run.

```sh
PYTHON=.venv/bin/python bash scripts/benchmark.sh cpu results-cpu
```

The script sets one OpenMP thread and one MKL thread. It measures 128 and 513 tokens with depth 64. The longer case exercises partial tiles on the CUDA backend. Each case uses a fixed seed, five warmup calls, and 20 timed calls.

For a CUDA experiment, use a supported Linux host with CUDA-enabled PyTorch and a compatible NVIDIA driver.

```sh
python -m pip install '.[gpu,profile]'
python -c 'import torch, triton; print(torch.cuda.get_device_name())'
make check
PYTHON=.venv/bin/python bash scripts/benchmark.sh cuda results-cuda
```

Inspect the parity test result before comparing GPU performance. The GPU suite runs both PyTorch and Triton with FP16 on the same host.

| Artifact | Interpretation |
|---|---|
| `trials.jsonl` | Each line contains one case. Times use nanoseconds. Group by kernel, token count, dtype, and backend. |
| `revision.txt`, `packages.txt`, `python.txt`, `platform.txt` | These files identify the software and platform used for the experiment. |
| `nvidia-smi.txt` | The CUDA suite records GPU and driver details. |
| `ceilings.json` | Device-copy bandwidth uses bytes per second. Matrix multiplication throughput uses FLOP/s. |
| `roofline.svg` | The horizontal axis uses modeled FLOP per minimum byte. The vertical axis uses modeled GFLOP/s. |

## Diagnose a failed run

| Symptom | Check |
|---|---|
| CUDA parity skipped | Check `torch.cuda.is_available()` and the Triton import in the test interpreter. |
| Triton rejects tensors | Inspect device, layout, dtype, head depth, and channel count. [Tensor contracts](system-design.md#tensor-contracts) |
| Roofline rejects records | Use CUDA records from the calibration GPU model with one common dtype. |
| Benchmark directory exists | Choose a new directory to preserve the earlier samples. |
| CUDA workflow remains queued | Check the self-hosted Linux x64 runner and its `cuda` label. Start the workflow from `main`. |

## Build and profile the Zig runtime

1. Install Zig 0.16.0.
2. Run `make native-check`.
3. Run `make native`.
4. Run `zig-out/bin/kernel-bench-zig --tokens 513 --depth 64 --repeats 20`.
5. Run `kernel-bench --backend zig --tokens 513 --depth 64 --repeats 20` to measure the Python boundary.

The standalone benchmark uses deterministic FP32 inputs, one group, and reusable buffers. Each timing record includes nanoseconds, modeled operations, and minimum bytes. It profiles CPU kernels. Use the CUDA workflow for GPU roofline calibration.

A wheel contains the shared library. An unpackaged checkout looks for the library under `zig-out`. The Python build system installs the pinned Zig compiler in its isolated build environment.

## Verify standalone CUDA

Use Linux with an NVIDIA GPU of compute capability 7.5 or later and a driver that accepts PTX 6.3. The benchmark selects device zero in the visible device set.

1. Build the native executable.
2. Run `zig-out/bin/kernel-bench-zig --device cuda --tokens 67 --depth 17 --repeats 3`.
3. Check that both JSON records name the physical GPU.
4. Run the larger workload from the README.

The native GPU benchmark uses one batch and one head. Select a depth from 1 through 128. Each run compares results with the CPU implementation before it writes timing records. Linux CI assembles the PTX without executing it. The hardware workflow executes the kernels.

The Python GPU backend uses Triton for its FP16 and BF16 autotuned kernels. The standalone FP32 path uses Zig for both host and device code.
