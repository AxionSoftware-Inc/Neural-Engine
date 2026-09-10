# P-003 native stable-prefix staged-growth audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Base recipe: 300M ordered factorized bank + rank-8 step adapter + 25% low/high
edge mix

## Question

Can a larger native bank preserve the parent's virtual circuit semantics and
then expose new capacity without causing the route fragmentation seen in the
ordinary 500M run?

## Architecture change

`factor_address_layout=stable_prefix` keeps the 300M factor-pair mapping in
the first 22,800 virtual addresses. The 500M bank uses 197 factor rows; new
factor pairs are appended after the parent's address set. The factorized
router and circuit bank share this same address map.

The checkpoint expander copies reusable factor rows and, for the stable layout,
copies the learned per-address mix for the parent prefix. The new sixth router
level is initialized separately. No active-circuit budget or circuit body
formula was changed.

## Protocol

- RTX 3060, CUDA, batch 128, AdamW, same two seeds 17/18;
- parent checkpoints are the 300M two-edge-mix 10k checkpoints;
- stage 1: 500M bank, but `routing_capacity=22800` and `routing_depth=5`,
  3,000 steps;
- stage 2: same checkpoint, then full 500M capacity/depth, 3,000 steps;
- 24-batch evaluator: uniform, combination-holdout, low-edge `[0,7]`, and
  high-edge `[56,63]`.

## Main result

| Condition | 300M continuation, mean | 500M stable-prefix stage 2, mean | Delta |
|---|---:|---:|---:|
| Uniform exact accuracy | 80.920% | **82.040%** | **+1.120 pp** |
| Uniform hard-task mean | 54.796% | **56.695%** | **+1.899 pp** |
| Combination holdout | 81.419% | **82.357%** | **+0.938 pp** |
| Combination hard-task mean | 55.154% | **57.107%** | **+1.953 pp** |
| Low-edge exact accuracy | 97.079% | **97.617%** | **+0.538 pp** |
| High-edge exact accuracy | 97.148% | **97.522%** | **+0.373 pp** |

Mean uniform dead-circuit fraction fell from `24.17%` in the 300M control to
`14.21%` after the full-capacity stage. The stage-1 number is not comparable
because the extra 15,800 500M rows are intentionally unreachable during the
warm-up.

## Interpretation

This is the strongest native capacity signal so far. Stable address semantics
plus staged exposure removed the earlier route-fragmentation failure and gave
a reproducible two-seed quality improvement on both ordinary and difficult
probes.

It is not yet a final scaling claim: stage 2 has received 6,000 additional
steps in total, while the comparison shown above has only a 3,000-step 300M
continuation. A 6,000-step 300M continuation is required to isolate capacity
from extra optimization time. The 500M stable-prefix variant therefore remains
`PROMISING OPT-IN — VALIDATION OPEN`, not the default and not a reason to jump
to 700M/1B yet.

## Controls and raw evidence

- The non-stable 500M warm-start improved route usage but did not beat a
  300M continuation under the available 3k continuation comparison.
- [Stable-prefix staged JSON](diagnostic_native_stable_prefix_staged_20260910.json)
- [Stable-prefix initial JSON](diagnostic_native_stable_prefix_initial_20260910.json)
- [Warm-start versus 300M continuation JSON](diagnostic_native_warmstart_vs_continued_20260910.json)
- [Warm-start initial JSON](diagnostic_native_warmstart_initial_20260910.json)
- [Stable-prefix full config](../configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_stable_prefix.yaml)
- [Stable-prefix warm-up config](../configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_stable_prefix_warmup.yaml)

Reproduction commands:

```powershell
python audit_native_ood.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_stage2_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_stage2_s18_3000.pt `
    results/checkpoints/ne500_stable_prefix_stage1_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_stage1_s18_3000.pt `
  --batches 24 `
  --output results/diagnostic_native_stable_prefix_staged_20260910.json
```
