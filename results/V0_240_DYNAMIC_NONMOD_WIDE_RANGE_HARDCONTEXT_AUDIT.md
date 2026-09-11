# V0.240 — hard digit context on clean wide-support arithmetic

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

## Question

V0.235 showed that straight-through hard digit context helps subtract on the
confounded above-range screen but does not solve multiply. V0.239 removes the
product-range OOD confounder by training and evaluating operands `0..95` while
holding out only depths; multiply becomes non-zero but depth-4 remains weak.
This screen asks whether hard cross-digit context provides a real carry/codec
benefit on that clean protocol.

## Protocol

- rank-16 cross-digit interaction with `straight_through_hard` context;
- matched base rank-128 no-interaction control from V0.239;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- training and evaluation operands `0..95`;
- train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, compact factorized evaluator;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- primary readout: operation-wise multiply and depth-4 accuracy; add/subtract
  are regression controls.

This is a clean depth-composition test, not an above-range value test and not
evidence for 700M/1B scaling.

## Results

Pending completion.
