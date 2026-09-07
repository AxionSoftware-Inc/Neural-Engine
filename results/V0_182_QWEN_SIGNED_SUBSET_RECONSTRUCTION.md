# V0.182 Qwen signed-subset reconstruction audit

Status: `REJECTED FOR ADOPTION`

## Savol

The hard sparse child previously used a fixed `E/K` scale and softmax weights
for the selected groups.  This experiment fits a signed coefficient vector for
every possible `K`-group subset:

```text
ŷ(S, x) = sum(e in S) c[S, e] * group_e(x)
```

The coefficients are fitted with a small ridge regression against the copied
teacher FFN output.  The hard path still evaluates only the selected groups;
there is no dense decoder and no extra neuron evaluation.  The exact-oracle
variant selects the subset using the new signed reconstruction cost, so it is
an upper-bound diagnostic rather than a deployment result.

## Protocol

Qwen3-0.6B, float32 CUDA, contiguous copied groups, `E=8`, `K=4`, grouped
dispatch, calibration from `data/qwen_calibration.txt`, and held-out evaluation
on `data/qwen_eval.txt`.  The coefficient fit uses the four calibration
batches.  The quality gate is fully active sparse-vs-teacher held-out
`alpha=0 CE delta <= +0.05`.

## Results

| Variant | Layers | Seed | Held-out +CE | Teacher top-1 | Local MSE | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| Signed subset, exact cost oracle | 25--26 | 2026 | `+0.0374` | `90.82%` | `0.786 / 0.924` | PASS |
| Signed subset, exact cost oracle | 25--26 | 2027 | `+0.0374` | `90.82%` | `0.786 / 0.924` | PASS |
| Signed subset, learned subset router, 100 steps | 25--26 | 2026 | `+0.0720` | `88.72%` | `0.960 / 1.073` | FAIL |
| Signed subset, learned router, 300-step soft target | 25--26 | 2026 | `+0.0660` | `88.82%` | `0.958 / 1.063` | FAIL |
| Signed subset, group-energy router, 300 steps | 25--26 | 2026 | `+0.1088` | `86.87%` | `1.304 / 1.266` | FAIL |
| Signed subset, pairwise cost router, 300 steps | 25--26 | 2026 | `+0.0659` | `88.62%` | `1.006 / 1.093` | FAIL |
| Signed subset, router hidden 512, 300 steps | 25--26 | 2026 | `+0.0642` | `88.53%` | `0.980 / 1.074` | FAIL |
| Signed subset, exact cost oracle | 23--26 | 2026 | `+0.0435` | `89.55%` | `0.601 / 0.613 / 0.733 / 0.879` | PASS |
| Signed subset, learned subset router, 100 steps | 23--26 | 2026 | `+0.0768` | `85.35%` | `0.819 / 0.760 / 0.848 / 0.956` | FAIL |

The exact-oracle result is reproducible, but it is not a large gain over the
existing contiguous-group exact-oracle control.  The learned router remains
about `+0.064--0.072` CE worse despite using the signed cost labels.  Enlarging
the router to 512 hidden units, using group-energy features, or using the
pairwise cost head did not close the oracle gap.

## Interpretation and decision

The fixed output scale is not the only problem, but replacing it with static
subset-specific signed coefficients does not solve the sparse Qwen transfer
problem.  The oracle proves that a coefficient contract can be numerically
stable at two and four layers; it does not prove that the route can be learned
from the available hidden-state signal.  Since the learned route fails and the
oracle is not better than the existing exact-subset control, the variant is
rejected as a default architecture.

This test covered signed reconstruction on the existing disjoint groups.  A
true overlapping/codebook decomposition remains a separate hypothesis; the
previous random-overlap screen was not reproducible and must not be reused as
evidence.  No 700M/1B run is justified by this result.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `tests/test_signed_subset.py`
- `results/runs/qwen_signed_subset_oracle_2layers_seed2026.json`
- `results/runs/qwen_signed_subset_oracle_2layers_seed2027.json`
- `results/runs/qwen_signed_subset_oracle_4layers_seed2026.json`
- `results/runs/qwen_signed_subset_router_2layers_seed2026.json`
- `results/runs/qwen_signed_subset_router_4layers_seed2026.json`
- `results/runs/qwen_signed_subset_router300_soft_2layers_seed2026.json`
- `results/runs/qwen_signed_subset_router300_energyinput_2layers_seed2026.json`
- `results/runs/qwen_signed_subset_pairwise_router300_2layers_seed2026.json`
- `results/runs/qwen_signed_subset_router512_2layers_seed2026.json`

## Reproduction

Exact-oracle two-layer run:

```powershell
python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 4 --partition-mode contiguous `
  --route-source oracle-subset --calibration-rank 1 `
  --calibration-mode signed-subset --dispatch-mode grouped `
  --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --train-batches 4 --eval-batches 2 `
  --batch-size 8 --sequence-length 128 --steps 0 --hard-train-steps 0 `
  --router-supervision-steps 0 --router-target subset --seed 2026 `
  --output results/runs/qwen_signed_subset_oracle_2layers_seed2026.json
```
