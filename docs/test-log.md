# Kernel verification log

## Native implementation at d4b9a00

The local checks below ran on 2026-11-27 with macOS arm64, Zig 0.16.0, Python 3.12.12, and PyTorch 2.14.0. [Source revision](https://github.com/asigdel29/triton-kernel-optimizer/commit/d4b9a00e71084d2cccd947c4c91c6ce33dca4a56)

| Command | Observed outcome |
| --- | --- |
| `make native-check` | Zig format verification and the native test command passed. |
| `make native-bench ARGS="--tokens 17 --depth 13 --repeats 3"` | Release build passed. Both kernels wrote three FP32 samples on aarch64. |
| `make check PYTHON=.venv/bin/python` | Five Python tests passed. One test skipped because the host lacked CUDA and Triton. |
| `make demo PYTHON=.venv/bin/python` | PyTorch emitted one FP32 CPU record for each kernel at 17 tokens and depth 64. |

The native smoke run reported 7,956 modeled operations for attention and 1,105 for RMSNorm. These values verify the requested dimensions and operation counts. Compare latency only with a matched run on the same host.

## Hosted build evidence

Both Python matrix jobs passed for this revision. The 3.12 job assembled the two CUDA entry points and collected the CPU artifact. [GitHub Actions run](https://github.com/asigdel29/triton-kernel-optimizer/actions/runs/46058068975)

GPU execution, numerical parity on a device, and measured GPU ceilings require a CUDA benchmark run. No device measurements accompany this entry.
