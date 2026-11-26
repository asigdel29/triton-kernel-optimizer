# Graph Report - triton-kernel-optimizer  (2026-11-26)

## Corpus Check
- 5 files · ~1,384 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 59 nodes · 94 edges · 8 communities (1 shown, 7 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- bench.py
- residual_rms_norm
- KernelTests
- __init__.py
- _attention
- _rms

## God Nodes (most connected - your core abstractions)
1. `residual_rms_norm()` - 13 edges
2. `causal_attention()` - 9 edges
3. `_attention()` - 9 edges
4. `KernelTests` - 7 edges
5. `rms_norm()` - 6 edges
6. `_rms()` - 5 edges
7. `_check_inputs()` - 4 edges
8. `main()` - 4 edges
9. `measure()` - 4 edges
10. `main()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `residual_rms_norm()` --calls--> `rms_norm()`  [EXTRACTED]
  kernel_optimizer/__init__.py → kernel_optimizer/gpu.py
- `main()` --calls--> `causal_attention()`  [EXTRACTED]
  kernel_optimizer/bench.py → kernel_optimizer/__init__.py
- `main()` --calls--> `residual_rms_norm()`  [EXTRACTED]
  kernel_optimizer/bench.py → kernel_optimizer/__init__.py

## Import Cycles
- None detected.

## Communities (8 total, 7 thin omitted)

### Community 1 - "residual_rms_norm"
Cohesion: 0.21
Nodes (4): main(), causal_attention(), _check_inputs(), residual_rms_norm()

## Knowledge Gaps
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `residual_rms_norm()` connect `residual_rms_norm` to `bench.py`, `KernelTests`, `__init__.py`, `test_kernels.py`?**
  _High betweenness centrality (0.299) - this node is a cross-community bridge._
- **Why does `_attention()` connect `_attention` to `__init__.py`, `_rms`, `gpu.py`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Why does `causal_attention()` connect `residual_rms_norm` to `bench.py`, `KernelTests`, `__init__.py`, `test_kernels.py`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._