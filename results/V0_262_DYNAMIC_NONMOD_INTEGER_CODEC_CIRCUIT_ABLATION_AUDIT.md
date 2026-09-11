# V0.262 — integer codec circuit ablation

**Date:** 2026-09-11  
**Status:** **DIAGNOSTIC COMPLETE; NO CIRCUIT CONTRIBUTION DETECTED**

## Question

V0.260–V0.261 reach nearly perfect quality with the exact integer state and
full-range codec, but the learned sparse body and router are still present in
the checkpoint. This paired ablation asks whether the circuit residual changes
the result at all.

The same generated batches are evaluated twice: once with the normal circuit
residual scale `1.0`, then with residual scale `0.0`. The exact integer state,
output decoder, digit head, and all other parameters remain unchanged.

## Protocol

- checkpoints: V0.260 seed17 and V0.261 seed18
- held-out values: `0..95`
- depths: `3..4`
- examples per depth: `256`
- paired RNG state: identical batch for enabled/disabled passes
- no training and no weight changes

## Results

| Seed | Operation | Circuit on | Circuit off | Delta |
|---:|---|---:|---:|---:|
| 17 | All operations | 100.0000% | 100.0000% | 0.0000 pp |
| 17 | Add | 100.0000% | 100.0000% | 0.0000 pp |
| 17 | Subtract | 100.0000% | 100.0000% | 0.0000 pp |
| 17 | Multiply | 100.0000% | 100.0000% | 0.0000 pp |
| 18 | All operations | 99.4141% | 99.4141% | 0.0000 pp |
| 18 | Add | 100.0000% | 100.0000% | 0.0000 pp |
| 18 | Subtract | 100.0000% | 100.0000% | 0.0000 pp |
| 18 | Multiply | 99.8047% | 99.8047% | 0.0000 pp |

## Decision and interpretation

This confirms that the V0.255–V0.261 quality jump is a terminal exact-codec
control result. The sparse circuit/router path is computationally irrelevant
to the measured output in this configuration because the all-operation exact
head reads the separately maintained algebraic integer state.

This does **not** justify deleting circuits from the Neural Engine: the probe
only covers the synthetic arithmetic task and an opt-in exact-prior branch.
It does mean that these numbers cannot be used as evidence of learned sparse
reasoning or as justification for 700M/1B scaling.

The next research gate remains a prior-free task/dataflow benchmark where the
output cannot bypass the learned circuit state.

## Artifact

- probe: `probe_integer_codec_circuit_ablation.py`
- raw report: `results/runs/v0_262_integer_codec_circuit_ablation.json`
