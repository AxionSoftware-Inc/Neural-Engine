# P-001 strided M=8 retrieval schedule — seed17/18 paired audit

**Decision: REJECTED**

The patch is opt-in and changes candidate retrieval schedule only. M=8,
active=2, the existing key-score selector, circuit body, recurrent state
update, correction path, and default model are unchanged.

## Quality

| Seed | Base CE | Patch CE | Δ CE | Base acc | Patch acc | Δ acc | Dead | Latency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 2.4459 | 2.8685 | +0.4226 | 48.18% | 41.35% | -6.82 pp | 0/32 | 0.816x |
| 18 | 2.6269 | 2.7669 | +0.1401 | 46.17% | 41.61% | -4.56 pp | 0/32 | 1.195x |

## Retrieval vs selection

| Seed | Variant | Recall | Retrieval mean | Retrieval p95 | Selection mean | Selection p95 | Selector CE | Selector acc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | baseline | 9.9% | 0.1953 | 0.9801 | 0.2624 | 1.4184 | 2.5044 | 47.3% |
| 17 | patched | 4.2% | 0.2484 | 1.0516 | 0.3677 | 1.5863 | 2.8838 | 42.5% |
| 18 | baseline | 12.1% | 0.1710 | 0.8565 | 0.2759 | 1.4596 | 2.4634 | 46.9% |
| 18 | patched | 4.3% | 0.2258 | 0.8601 | 0.3308 | 1.5248 | 2.7270 | 44.2% |

## Active cost

The schedule adds no trainable parameter and no probe forward. Candidate key
rows remain M=8 and executed circuit rows remain active=2. Measured latency
is still gated separately because Python hook overhead is real experiment cost.

## Acceptance gate

- PASS — `seed17_18_protocol`
- PASS — `full_5000_step_checkpoint`
- FAIL — `candidate_recall_non_decreasing`
- FAIL — `mean_p95_retrieval_regret_reduction_gte_10pct`
- FAIL — `mean_p95_selection_regret_reduction_gte_10pct`
- FAIL — `mean_hard_accuracy_delta_gte_2pp`
- PASS — `dead_circuits_lte_3`
- PASS — `latency_lte_1_25x`
- PASS — `active_cost_unchanged`

A failed gate means REJECTED for adoption. CE-only improvement is explicitly
insufficient. See the JSON for by-step regret and complete active-cost fields.

## Reproduction

```bash
python benchmark_p001_strided_retrieval.py \
  --checkpoint /path/to/seed17.pt \
  --checkpoint /path/to/seed18.pt \
  --device cuda \
  --output results/runs/p001_strided_retrieval_s17_s18.json \
  --markdown results/P001_STRIDED_RETRIEVAL_S17_S18.md
```
