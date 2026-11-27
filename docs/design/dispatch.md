# Dispatch and precision

The public Python functions validate all input tensors before dispatch. Each call returns a new tensor with the activation shape, dtype, and device.

| Request | Execution path | Input restrictions |
| --- | --- | --- |
| Python `auto` | Zig on CPU when the shared library loads. PyTorch otherwise. | All tensors share a device and dtype. |
| Python `zig` | Zig CPU library through the C ABI | FP32, FP16, or BF16 CPU tensors. The adapter makes contiguous copies when needed. |
| Python `torch` | PyTorch reference operations | FP32, FP16, or BF16 on a device that supports the operation. |
| Python `triton` | Triton on CUDA | Contiguous FP16 or BF16 tensors. Attention depth is 16, 32, 64, or 128. RMSNorm depth is at most 16,384. |
| Native executable | Zig CPU or embedded Zig PTX | FP32, one batch, one head. The CUDA benchmark accepts depths from 1 through 128. |

## Buffer ownership

The Python adapter owns the temporary contiguous inputs and output. It provides a private FP32 scratch row for each CPU attention call. The C ABI borrows those buffers until return.

The executable initializes device buffers before its CUDA timing loop. Its context owns the device allocations. After each case, the executable copies the result back and checks it against the CPU kernel.

## Shape decisions

Attention takes three equal tensors with layout `[batch, heads, tokens, depth]`. Causality restricts each output row to keys at or before that row. RMSNorm takes equal activation and residual shapes. Its weight length must match the final axis. The public API checks that epsilon is positive and finite.

For the online softmax recurrence and launch sequence, see the [system design](../system-design.md).
