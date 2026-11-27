# Graph Report - triton-kernel-optimizer  (2026-11-27)

## Corpus Check
- 12 files · ~4,145 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 124 nodes · 194 edges · 8 communities (6 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- __init__.py
- causal_attention
- cuda.zig
- bench.py
- _attention
- setup.py
- kernels.zig
- device.zig

## God Nodes (most connected - your core abstractions)
1. `residual_rms_norm()` - 13 edges
2. `causal_attention()` - 11 edges
3. `_attention()` - 9 edges
4. `KernelTests` - 9 edges
5. `attention()` - 6 edges
6. `rms_norm()` - 6 edges
7. `attention_f32()` - 6 edges
8. `_check_inputs()` - 5 edges
9. `_rms()` - 5 edges
10. `rms_norm()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `causal_attention()` --calls--> `attention()`  [EXTRACTED]
  kernel_optimizer/__init__.py → kernel_optimizer/zig.py
- `main()` --calls--> `residual_rms_norm()`  [EXTRACTED]
  kernel_optimizer/bench.py → kernel_optimizer/__init__.py
- `_check_inputs()` --calls--> `available()`  [EXTRACTED]
  kernel_optimizer/__init__.py → kernel_optimizer/zig.py
- `main()` --calls--> `causal_attention()`  [EXTRACTED]
  kernel_optimizer/bench.py → kernel_optimizer/__init__.py
- `residual_rms_norm()` --calls--> `rms_norm()`  [EXTRACTED]
  kernel_optimizer/__init__.py → kernel_optimizer/zig.py

## Import Cycles
- None detected.

## Communities (8 total, 2 thin omitted)

### Community 0 - "__init__.py"
Cohesion: 0.12
Nodes (7): _check_inputs(), residual_rms_norm(), attention(), available(), _library(), _native(), rms_norm()

### Community 1 - "causal_attention"
Cohesion: 0.11
Nodes (3): main(), causal_attention(), KernelTests

### Community 2 - "cuda.zig"
Cohesion: 0.15
Nodes (8): cpu(), main(), Case, attention, rms_norm, check(), Driver, Run

### Community 4 - "_attention"
Cohesion: 0.21
Nodes (3): _attention(), _rms(), rms_norm()

### Community 6 - "kernels.zig"
Cohesion: 0.39
Nodes (7): attention(), BF16, ko_attention(), ko_rms_norm(), load(), rmsNorm(), store()

### Community 7 - "device.zig"
Cohesion: 0.57
Nodes (7): attention_f32(), blockId(), exponential(), invSqrt(), laneId(), rms_f32(), warpSum()

## Knowledge Gaps
- **4 isolated node(s):** `attention`, `rms_norm`, `Driver`, `BF16`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 53 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `residual_rms_norm()` connect `__init__.py` to `causal_attention`, `bench.py`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **What connects `attention`, `rms_norm`, `Driver` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1164021164021164 - nodes in this community are weakly interconnected._
- **Why does `causal_attention()` connect `causal_attention` to `__init__.py`, `bench.py`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Should `causal_attention` be split into smaller, more focused modules?**
  _Cohesion score 0.11255411255411256 - nodes in this community are weakly interconnected._
- **Why does `_attention()` connect `_attention` to `__init__.py`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._