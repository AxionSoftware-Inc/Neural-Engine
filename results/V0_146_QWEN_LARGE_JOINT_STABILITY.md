# V0.146 Qwen Large-Block Joint Stability

## Question

Does full-model joint refinement repair the two-layer composition failure at
the larger 8×128 token block, and does making training use the final hard route
solve the problem?

## Protocol

The two adjacent layer-25/26, normless, four-expert top-2 grouped bank is
started from the same 300-step local distillation used in V0.145. Seed 2026 is
then tested with 50 full-model joint steps at learning rates `1e-3` and
`1e-4`, followed by a hard-route training variant at `1e-3`. Teacher logits
are offloaded to CPU float16 so the 12GB GPU does not page under the 8×128
workload. The combined `shared_alpha_0` path is evaluated on the held-out
set.

## Results

| joint mode | final joint loss | combined alpha=0 CE delta | quality gate |
|:---|---:|---:|:---:|
| soft route, lr 1e-3 | 3.874 | +0.1247 | fail |
| soft route, lr 1e-4 | 0.496 | +0.1017 | fail |
| hard route, lr 1e-3 | 3.593 | +0.1162 | fail |

The unrefined V0.145-style seed-2026 baseline was already a failure at this
block (`+0.0993`). Lowering the learning rate reduced the joint loss but did
not repair held-out composition. Hard-route training also failed despite
matching the final top-k execution rule.

## Interpretation

The short joint refinement signal from V0.145 is batch-regime dependent; it
does not generalize to this larger token block. The failure is not explained
solely by soft-vs-hard route mismatch or by GPU memory. Keeping more teacher
logits on CPU fixes the memory problem, but not the quality problem.

## Decision

**Reject large-block joint refinement as the current composition fix.** Keep it
as an optional small-batch research control, not as a reason to scale the
model. The more reliable path is a small interface correction trained with
each local child, tested next at minimal rank. The teacher-logit CPU offload is
kept because it makes future experiments memory-safe.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_joint50_b8s128_seed2026.json`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_joint50_lr1e-4_b8s128_seed2026.json`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_hardjoint50_b8s128_seed2026.json`
