# Triton kernel optimizer

Measure causal attention and residual RMSNorm with the same inputs across backends.
The Python API supports forward inference.
The attention kernel keeps a running softmax denominator per query row.

## Quickstart

Use Python 3.10 or later and PyTorch 2.3 or later.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
make check
make demo
make bench
```

For a Linux host with a supported NVIDIA GPU, install the GPU extra.

```sh
python -m pip install '.[gpu]'
make check
make bench ARGS="--device cuda --backend triton --dtype float16 --tokens 1025 --depth 64"
```

## Interface

Import `causal_attention` and `residual_rms_norm` from `kernel_optimizer`.
Set `backend="triton"` to select GPU execution.
Use equal contiguous input tensors for the GPU backend.
Attention uses the shape [batch, heads, tokens, depth].
RMSNorm accepts one weight per channel and returns the normalized sum.
Both functions preserve the input dtype.

## Measurements

Each output line contains one JSON record with the raw samples in nanoseconds.
The timer includes launch and synchronization costs on CUDA.
The traffic model counts each input read and each output write once.
Attention operation counts include both causal matrix products.
The RMSNorm estimate counts five operations per channel.
The estimates exclude softmax, scalar row operations, and cache effects.
Use hardware counters for actual traffic.
Install the profile extra to generate a roofline plot on the measurement GPU.

```sh
python -m pip install '.[profile]'
kernel-bench --backend triton --device cuda --dtype float16 > gpu.jsonl
python -m kernel_optimizer.roofline gpu.jsonl roofline.svg
```

The calibrator times device copies and square matrix products.
Use one GPU model and precision per plot.
Empirical GPU results require a CUDA run.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Continuous integration and benchmark artifacts

The CI workflow builds the wheel and tests it on Python 3.10 and 3.12. Each push and pull request produces a CPU benchmark artifact.

Run the same CPU benchmark locally from the repository root.

```sh
PYTHON=.venv/bin/python bash scripts/benchmark.sh cpu benchmark-results
```

Select a fresh output directory for each run. The suite uses seed zero, one CPU thread, five warmups, and 20 recorded calls. It covers 128 and 513 tokens at head depth 64. Compare raw samples only on equivalent hosts.

For GPU results, register a Linux x64 runner with the `cuda` label. Install a compatible NVIDIA driver, Python, and CUDA-enabled PyTorch on the runner. Start **CUDA benchmark** on the main branch from GitHub Actions. It tests GPU parity, measures both backends, and uploads the trials, calibration ceilings, environment, and roofline plot.

## Documentation

- [System design and UML diagrams](docs/system-design.md)
- [Operating guide and benchmark artifacts](docs/operations.md)
- [Repository instructions](AGENTS.md)

## Graphify source map

Read the [source map report](graphify-out/GRAPH_REPORT.md) or query the [graph data](graphify-out/graph.json). Open the [interactive map](graphify-out/graph.html) locally in a browser.

Install Graphify 0.9.76 with `uv tool install graphifyy==0.9.76`. Run these commands from the repository root.

```sh
make graph
graphify query "rms_norm"
```

The build reads tracked Python files from the working tree, including tests. Stage new Python files before you refresh the map. Graphify uses local syntax analysis with model backends disabled. The system design diagrams describe runtime behavior.
