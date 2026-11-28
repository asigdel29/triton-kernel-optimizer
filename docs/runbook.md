# Operator runbook

## Build and smoke-test on CPU

1. Install Zig 0.16.0.
2. Run `make native-check` at the repository root.
3. Run `make native`.
4. Run the command below.

```sh
zig-out/bin/kernel-bench-zig --tokens 17 --depth 13 --repeats 3
```

Expect attention and RMSNorm records. Both must identify the CPU, FP32 inputs, and three samples.

## Verify the Python wheel

1. Complete the [Python quickstart](../README.md#quickstart).
2. Activate the same virtual environment.
3. Run the installed command from outside the checkout.

```sh
(cd /tmp && kernel-bench --backend zig --tokens 17 --repeats 2)
```

The output must name the Zig backend. A missing library in the checkout can indicate that the wheel and source use different import paths. The [operations reference](operations.md) explains library resolution.

## Collect CPU or GPU artifacts

1. Select a fresh output directory.
2. Use the first command below for the CPU suite.
3. On a CUDA host, install the GPU and profile extras.
4. Use the remaining commands on that host.

```sh
PYTHON=.venv/bin/python bash scripts/benchmark.sh cpu results-cpu
python -m pip install '.[gpu,profile]'
python -c 'import torch, triton; print(torch.cuda.get_device_name())'
make check
PYTHON=.venv/bin/python bash scripts/benchmark.sh cuda results-cuda
```

Inspect the test output for skipped GPU checks. The CUDA suite writes raw trials, calibration ceilings, GPU metadata, and a roofline SVG. Keep the entire directory with the report.

## Check the native CUDA path

Build the native executable on Linux. Run `zig-out/bin/kernel-bench-zig --device cuda --tokens 67 --depth 17 --repeats 3`. Each record follows a comparison between the CUDA result and the CPU reference.

Hosted CI can assemble the device code. To exercise dispatch on hardware, start **CUDA benchmark** from `main` with a registered Linux x64 runner labeled `cuda`. See the [operations reference](operations.md#diagnose-a-failed-run) for errors and recovery.
