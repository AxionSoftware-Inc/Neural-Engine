# V0.259 — depth-3 unseen-value extrapolation

**Date:** 2026-09-11  
**Status:** **PASSED AS VALUE-COVERAGE VALIDATION**

## Question

The V0.255 full-range codec fixed unseen operand `96` at depth 4. A direct
`96..127` depth-4 screen is invalid for the current `8.59B` class space,
because five-operand products can exceed that classifier range. This probe
therefore evaluates depth 3 only, where four-operand products through `127`
remain legal.

This is a value-extrapolation validation of V0.255, not a new training run.

## Protocol

- checkpoints: V0.255 seed17 and seed18
- operands: unseen `96..127`
- depth: exactly `3` operations / `4` operands
- examples: `256`
- no weights changed
- targets remain inside `num_classes=8,589,934,592`

## Results

| Seed | All ops | Add | Subtract | Multiply |
|---:|---:|---:|---:|---:|
| 17 | 96.4844% | 100.0000% | 100.0000% | 100.0000% |
| 18 | 96.0938% | 100.0000% | 100.0000% | 100.0000% |
| **Mean** | **96.2891%** | **100.0000%** | **100.0000%** | **100.0000%** |

## Decision

**Passed as a value-coverage validation.** V0.255 is not merely memorizing
the training range `0..95`: every individual operation is exact on the
unseen `96..127` depth-3 screen. The remaining aggregate errors come from
mixed-operation programs, not the operation-specific terminal codec.

This strengthens the diagnosis that the major bottleneck in the synthetic
benchmark was numeric readout/range coverage. It still does not prove
general-purpose learned arithmetic, because the exact algebraic state provides
the known add/subtract/multiply semantics.

## Artifact

- probe: `probe_depth3_value_extrapolation.py`
- raw report: `results/runs/v0_259_fullrange_depth3_unseen_values_96_127.json`
