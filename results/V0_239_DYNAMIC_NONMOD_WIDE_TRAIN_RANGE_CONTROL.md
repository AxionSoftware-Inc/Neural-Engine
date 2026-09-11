# V0.239 — wide training-range multiply control

**Date:** 2026-09-11  
**Status:** `VALIDATED; PRODUCT-RANGE OOD CONFIRMED`

## Question

The corrected above-range protocol trains on operands `0..63` and evaluates
on `64..95`. V0.236–V0.238 show that multiply remains at `0%` while add and
subtract can generalize. Before changing the recurrent state again, this
control asks whether multiply is simply unsupported because products from
`64..95` (up to `9025`) never occur during training.

## Protocol

- base four-digit rank-128 no-interaction model; no new architecture;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- training operands `0..95`, evaluation operands `0..95`;
- train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, compact factorized evaluator;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- task-wise follow-up for add/subtract/multiply.

Interpretation:

- if multiply becomes non-zero, the main failure is product-range support/OOD;
- if multiply remains near zero, the recurrent composition or value-to-output
  dataflow is the stronger suspect;
- this result does not authorize 700M/1B scaling.

## Results

The two fresh runs completed with unchanged `7,477,191` total and
`2,178,032` active-estimate parameters. Aggregate held-out depth accuracy is
`78.9063%` / `79.4922%` across seeds17/18, with depth-4 `75.00%` / `72.66%`.
The task-wise diagnostic gives the following two-seed means:

| operation | held-out | depth-3 | depth-4 | CE |
|---|---:|---:|---:|---:|
| add | 100.00% | 100.00% | 100.00% | 0.002896 |
| subtract | 91.9922% | 97.2656% | 86.7188% | 0.593367 |
| multiply | 15.4297% | 23.2422% | 7.6172% | 29.589287 |
| all operations | 82.2266% | 86.3281% | 78.1250% | 2.744670 |

This is a clear separation from the `0..63 → 64..95` screen, where multiply
was `0%` in both arms. Seeing products from the wider range during training
restores non-zero multiply accuracy without adding capacity, confirming that
the earlier zero was substantially a product-range support/OOD confounder.
However, depth-4 multiply remains only `7.62%`, so the recurrent
composition/dataflow problem is not solved. This is a protocol correction and
a useful positive diagnostic, not a default architecture win.

## Decision

**RETAIN AS THE CLEANER EVALUATION PROTOCOL; DO NOT SCALE CAPACITY YET.**
Future architecture comparisons must either train and evaluate on the same
operand support with held-out depth, or explicitly label product-range OOD as
a separate generalization gate. The next experiment should target deep
multiply composition on this wide-support protocol, with add/subtract as
controls.
