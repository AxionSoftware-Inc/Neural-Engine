# V0.298/V0.299 — Leading-digit contract fix and high-value stress test

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `CONTRACT FIX ACCEPTED; QUALITY HYPOTHESIS STILL OPEN`

## V0.298 — contract correction

The eight-digit base-16 output has a 32-class leading head because
`2^33 / 16^7 = 32`. The old `factorized_digit_targets()` applied
`remainder(16)` to that leading target, aliasing classes `16..31` onto
`0..15`. The function now leaves the leading quotient intact and applies
`remainder(16)` only to trailing digits. A regression test covers leading
targets `0, 15, 16, 31`.

V0.298 retrained the V0.290 numeric bridge under the corrected contract for
5,000 steps, seeds `17/18`, with no other change.

| Seed | Held-out acc | CE | Depth 3 | Depth 4 | Train sec |
|---:|---:|---:|---:|---:|---:|
| 17 | 21.0449% | 12.1085 | 27.5391% | 14.5508% | 1116.9 |
| 18 | 19.9707% | 11.4341 | 25.7813% | 14.1602% | 1065.7 |
| **Mean** | **20.5078%** | **11.7713** | **26.6602%** | **14.3555%** | **1091.3** |

The hard metrics reproduce V0.290 exactly. Under the current protocol, train
depths are only `1..2`, whose products never reach leading classes `16..31`,
so the correction is necessary for correctness but supplies no new training
signal. It is now the canonical target/loss contract.

## V0.299 — high-value stress

The standard operation-wise screen uses random values and did not expose
leading classes `16..31`. V0.299 evaluates the corrected V0.298 checkpoints on
deterministic homogeneous programs with every input value in `80..95`, at
held-out depths `3` and `4`.

| Variant | Operation | Depth 3 acc | Depth 4 acc | Depth 3 CE | Depth 4 CE |
|---|---|---:|---:|---:|---:|
| V0.298 mean | add | 0.9766% | 0.0977% | 9.3798 | 11.1142 |
| V0.298 mean | subtract | 53.9063% | 1.7578% | 1.5140 | 10.9443 |
| V0.298 mean | multiply | **0.0000%** | **0.0000%** | **34.8429** | **44.5355** |

The high-value multiply failure is perfectly consistent across both seeds and
both held-out depths. This confirms that the normal random screen understated
the magnitude/depth problem. It does not prove that the leading head alone is
the cause: high-value multiply also creates the hardest recurrent products.

## Decision

The leading-digit target contract fix is accepted as a correctness fix, but it
does not change the quality frontier. V0.298 remains an opt-in diagnostic, not
a default or scaling signal. V0.299 closes the current random-screen blind
spot and motivates a controlled depth-coverage experiment: train on depths
`1..3` and evaluate depth `4` under the same wide value range. No 700M/1B run
follows yet.

## Artifacts

- `train_dynamic_composition.py`
- `benchmark_operationwise_checkpoints.py`
- `tests/test_operationwise_checkpoint_eval.py`
- `results/runs/v0_298_numeric_bridge_contractfix_seed17_5000.json`
- `results/runs/v0_298_numeric_bridge_contractfix_seed18_5000.json`
- `results/operationwise_fixed_checkpoint_eval_v0_298.json`
- `results/operationwise_high_value_v0_299.json`

## Reproduction

```powershell
python -m pytest -q
python -u benchmark_operationwise_checkpoints.py --checkpoint results/checkpoints/v0_298_numeric_bridge_contractfix_seed17_5000.pt --checkpoint results/checkpoints/v0_298_numeric_bridge_contractfix_seed18_5000.pt --examples-per-case 512 --value-min 80 --value-max 95 --device cuda --output results/operationwise_high_value_v0_299.json
```
