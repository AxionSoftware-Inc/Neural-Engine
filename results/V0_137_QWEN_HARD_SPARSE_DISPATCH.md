# V0.137 Qwen Hard Sparse Dispatch Audit

## Question

Does the routed Qwen child bank actually execute only the selected top-k
circuits, rather than evaluating the full bank and gathering afterward?

## Protocol

The V0.112 two-layer Qwen3-0.6B routed-child protocol is repeated with two
192-wide child banks, four experts per layer, and top-2 evaluation. Training
still uses the soft all-expert mixture for gradient coverage. Evaluation now
dispatches each token only to its selected expert bodies and accumulates their
weighted outputs. Thus the expected expert-body execution fraction is 2/4 =
50%. No wall-clock speedup is claimed; this is an execution-path audit.

## Results

| seed | alpha=0 two-child CE delta | teacher top-1 agreement | expected expert-body fraction | quality gate |
|---:|---:|---:|---:|:---:|
| 2026 | +0.0537 | 94.14% | **50%** | fail |
| 2027 | +0.0708 | 92.29% | **50%** | fail |

The quality values match the earlier full-evaluation implementation within
normal floating-point variation, so replacing full-bank evaluation with
selected-token dispatch did not create a hidden accuracy change. The two-layer
handoff remains above the `+0.05` CE gate, especially on seed 2027.

## Interpretation

The active-compute mechanism is now real in the research path: unselected
expert bodies receive no evaluation call for a token. This validates the
implementation needed for sparse execution, but the current child
representations still accumulate too much error across two adjacent layers.
The failure is therefore quality/representation drift, not a false sparsity
measurement.

## Decision

**Keep selected-token dispatch; reject the current two-layer routed handoff as
a quality GO.** Continue with the stable single-layer Qwen circuit-transfer
signal and the shared content-addressing interface. Do not claim production
latency savings until a grouped-kernel wall-clock benchmark is added.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_routed4x2_sparse_exec_seed2026.json`
- `results/runs/qwen_two_layer_routed4x2_sparse_exec_seed2027.json`
