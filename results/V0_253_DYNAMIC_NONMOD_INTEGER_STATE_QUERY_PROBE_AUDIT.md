# V0.253 — exact integer packet query-read probe

**Date:** 2026-09-11  
**Status:** **REJECTED FOR MAIN ARCHITECTURE ADOPTION**

## Question

V0.252's frozen exact-integer output overlay is the leading opt-in result.
This probe asks whether its exact integer packet also improves the recurrent
dataflow: inject the decoded packet into the recurrent query while leaving the
body, router, circuit bank, and trained overlay weights frozen. The only
variable is a fixed injection scale. This is an inference-only causal probe;
no new parameters are trained.

The hypothesis was that a numerically exact intermediate signal might help the
router/body on difficult multiply cases and make the quality gain more
architectural than a terminal readout patch.

## Protocol

- checkpoint: V0.252 codec-calibrated frozen overlay, seed17
- held-out values: `0..95`
- examples per depth: `256`
- seeds: `17` and `18`
- scales: `0.0`, `0.5`, `1.0`, `2.0`, `4.0`
- all model weights frozen; only `model.algebraic_integer_state_read_scale`
  changes at inference
- metrics: exact hard accuracy by operation and depth

The scale-0 row is the within-run control. Because this probe uses a fresh
evaluation draw, it should be compared within this table rather than mixed
with the taskwise V0.252 JSONs.

## Results

| Scale | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply | Mul d3 | Mul d4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 85.16% | 89.26% | 81.05% | 100.00% | 99.80% | 61.43% | 79.30% | 43.55% |
| 0.5 | 81.35% | 85.35% | 77.34% | 100.00% | 95.51% | 61.43% | 79.30% | 43.55% |
| 1.0 | 69.34% | 73.83% | 64.84% | 99.90% | 74.32% | 61.43% | 79.30% | 43.55% |
| 2.0 | 49.02% | 54.10% | 43.95% | 88.57% | 8.11% | 61.43% | 79.30% | 43.55% |
| 4.0 | 40.92% | 45.70% | 36.13% | 62.70% | 2.25% | 61.43% | 79.30% | 43.55% |

The table is the mean over seeds17/18. At scale 0.0 the seed-wise aggregate
was `84.9609% / 85.3516%`; at scale 0.5 it was `80.4688% / 82.2266%`.
Multiply was exactly unchanged at every scale in both seeds. The packet was
therefore not participating in the multiply decision path in this setup; it
only perturbed the shared learned path used by add/subtract and the aggregate
score.

## Decision

**Rejected as a quality or capacity fix.** The exact packet is useful as a
terminal multiply readout, but adding it to the shared recurrent query does
not create a useful intermediate circuit. It gives no multiply improvement
and has a clear negative interference threshold around scale `0.5` and above.
The default model and router remain unchanged.

This closes the narrow hypothesis that “the trained output packet can simply
be read by the existing recurrent query.” It does not prove that every
operation-conditioned or separately trained packet transition is impossible;
those would be different experiments. It does show that a raw additive
injection is the wrong interface.

## Consequences for the active problems

- **P-003 remains active:** the current positive result is a range-calibrated,
  multiply-specific output codec, not evidence of universal capacity scaling.
- **P-002 remains active:** the exact packet did not improve circuit selection
  or specialization because multiply was invariant across the query probe.
- **P-004 remains active:** no claim is made that this probe solves cascade
  credit assignment; the shared query perturbation was harmful.

The leading candidate remains V0.252 as an opt-in overlay. Before any 700M or
1B scale jump, the next useful experiment should keep the exact packet on the
multiply-only terminal path and test a separately normalized, operation-
conditioned transition or gate. A shared raw query residual should not be
repeated.

## Artifact

- probe: `probe_integer_state_read.py`
- raw report: `results/runs/v0_253_integer_state_read_probe_full.json`
- base sweep: `results/V0_249_252_DYNAMIC_NONMOD_INTEGER_CODEC_CALIBRATION_SWEEP_AUDIT.md`
