# Repository instructions

Read the [system design](docs/system-design.md) before changing dispatch, tensor contracts, or kernel arithmetic. Use the [operating guide](docs/operations.md) to verify results.

- Keep the CPU reference usable with PyTorch alone. Import Triton inside the GPU dispatch path.
- Preserve input validation, output shape, dtype, and device across supported backends.
- Compare numerical changes with the independent FP64 equations in `tests/test_kernels.py`.
- Run `make check` and `make demo` from the project environment.
- Use an odd token count and each supported head depth when you verify attention changes on CUDA.
- Report the hardware, precision, and test skips with benchmark evidence.
- Update the diagrams and operation counts when execution or measurement boundaries change.
- Write documentation in Simplified Technical English. Use one instruction per procedure step.

Use `graphify query "rms_norm"` to locate related code. Run `make graph` after source structure changes.
