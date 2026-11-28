# Measurement and roofline design

Use a separate result series for each timer boundary. A single host supplies all calibration data for one roofline.

| Instrument | Included in the timed region |
| --- | --- |
| Python benchmark | Dispatch, operation, output allocation, and CUDA synchronization when applicable. |
| Native CPU benchmark | Computation over preallocated FP32 buffers with a monotonic host clock. |
| Native CUDA benchmark | Kernel launch and device synchronization. Upload, context setup, and result verification occur outside the timer. |
| Roofline calibrator | Device work between CUDA events for a copy or a matrix product. |

## Model used in the chart

Let N denote token count and D denote head depth. The causal matrix products contribute `2 × N × (N + 1) × D` modeled operations. RMSNorm contributes `5 × N × D`. Minimum bytes count each input read and each output write once. Cache behavior, repeated reads, and softmax work affect the physical cost.

Arithmetic intensity equals modeled operations divided by minimum bytes. The calibrator uses a 256 MiB source for the copy and 4096-square matrices for compute. It takes the median of ten device-time samples after five warmups.

The roof at a given intensity is the lesser of measured compute throughput and intensity times measured bandwidth. The Python benchmark provides the throughput coordinates. Use its CUDA JSONL with the matching GPU and dtype when generating the SVG.

## Compare two runs

1. Check the source revision, device, packages, and dtype.
2. Select records with the same kernel, timer, and dimensions.
3. Retain individual latency samples with each reported statistic.
4. Use the artifact manifest files to explain configuration differences.

The [runbook](../runbook.md) gives the collection procedure.
