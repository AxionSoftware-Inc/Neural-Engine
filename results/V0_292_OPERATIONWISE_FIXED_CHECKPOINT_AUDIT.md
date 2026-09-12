# V0.292 — Fixed operation-wise checkpoint diagnostic

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `DIAGNOSTIC COMPLETE; NO DEFAULT CHANGE`

## Question

V0.290's numeric bridge was designed to help multiplication, while V0.291
showed that the value curriculum could hurt it. Before changing the model again,
this inference-only screen asks whether the quality gap is concentrated in
multiplication or is a general failure of the typed state.

## Protocol

- Existing checkpoints only; no training or weight updates;
- V0.288 typed-carry plain, V0.290 numeric bridge, and V0.291 numeric bridge
  plus curriculum;
- two checkpoints/seeds per variant (`17/18`);
- deterministic homogeneous programs: all `add`, all `subtract`, or all
  `multiply`;
- held-out depths `3` and `4`, values `0..95`, non-modular targets with the
  same target offset as training;
- 512 examples per operation/depth/seed, compact factorized evaluation;
- metrics are final hard accuracy and factorized CE.

## Results

| Variant | Operation | Depth 3 acc | Depth 4 acc | Depth 3 CE | Depth 4 CE |
|---|---|---:|---:|---:|---:|
| V0.288 typed plain | add | 67.8711% | 33.6914% | 1.6510 | 4.5706 |
| V0.288 typed plain | subtract | 65.2344% | 29.6875% | 1.7200 | 5.5010 |
| V0.288 typed plain | multiply | 3.6133% | 3.6133% | 24.3703 | 41.3643 |
| V0.290 numeric bridge | add | 72.4609% | 36.6211% | 1.9753 | 6.4173 |
| V0.290 numeric bridge | subtract | 67.9688% | 30.8594% | 1.5006 | 5.0514 |
| V0.290 numeric bridge | multiply | 3.9063% | 4.1992% | 23.8709 | 41.3290 |
| V0.291 bridge + curriculum | add | 74.3164% | 41.8945% | 1.0905 | 4.2642 |
| V0.291 bridge + curriculum | subtract | 71.2891% | 32.7148% | 1.8536 | 9.8807 |
| V0.291 bridge + curriculum | multiply | 3.2227% | 2.7344% | 20.2580 | 34.2518 |

The multiply path is the dominant failure for every checkpoint. V0.290's
numeric bridge changes multiply hard accuracy only from `3.6133%` to `3.9063%`
at depth 3 and from `3.6133%` to `4.1992%` at depth 4. V0.291 makes it worse:
`3.2227%` and `2.7344%`. In contrast, add and subtract reach roughly
`30–74%` depending on depth and checkpoint.

The result is not evidence that the router alone is at fault. The same sparse
router and typed state can carry useful add/subtract behavior, while the
multiply-specific dataflow cannot preserve the required cross-digit products
through held-out depth. The numeric bridge gave a small hard-accuracy movement
but did not create a reliable multiply circuit.

Because these are separately trained checkpoints rather than a single matched
weight-ablation run, the exact differences between variants are not causal
estimates. The robust conclusion is the operation gap itself: multiplication
needs a dedicated dataflow/training diagnostic before scaling or further router
complexity.

## Decision

No default or architecture adoption follows from V0.292. Keep V0.290 as the
leading opt-in candidate only. The next controlled test is a multiply-focused
training distribution on the same V0.290 body: if multiply improves strongly,
the bottleneck includes operation coverage; if it remains near chance, the
typed multiply transition is structurally insufficient. Add/subtract quality
must be measured separately and not mixed into that diagnostic conclusion.

## Artifacts

- `benchmark_operationwise_checkpoints.py`
- `tests/test_operationwise_checkpoint_eval.py`
- `results/operationwise_fixed_checkpoint_eval_v0_292.json`
- evaluated checkpoints under `results/checkpoints/v0_288_*`, `v0_290_*`, and
  `v0_291_*`

## Reproduction

```powershell
python -u benchmark_operationwise_checkpoints.py --examples-per-case 512 --device cuda --output results/operationwise_fixed_checkpoint_eval_v0_292.json
```
