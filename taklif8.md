# Taklif 8 — generalize the modular template interface

Status: **V0.55 positive on mod-64 and mod-32; general-purpose promotion still
blocked on non-arithmetic transfer**

## Current evidence

Random-init trainable templates reach 100% on unseen values and deeper programs
after 10k steps at both modulus 64 and modulus 32. The Dynamic Register
integration reaches 99.71% without circuit residual and 99.51% with it at the
same 10k budget. Therefore the modular interface, not sparse capacity, is the
current arithmetic solution.

## Next architecture

Separate the engine into two explicit paths:

```text
compact equivariant primitive interface -> exact/compositional state
learned sparse residual bank            -> task-specific correction/skill
```

The primitive interface must be configurable rather than hard-coded to one
benchmark. Operation token IDs should be renameable, modulus should be a
configuration, and initialization must be random for validation. The
residual bank should be optional and have a measurable contribution.

## Required tests

1. mod-16, mod-32, and mod-64 with two seeds;
2. renamed operation tokens and permuted primitive order;
3. a composition task with unknown/non-modular primitives;
4. no-residual, random-residual, and residual-initialized A/B;
5. active trainable parameters, latency, and memory; and
6. a small real-data control before any 300M/1B scaling.

## Stop conditions

- If performance depends on identity initialization, keep the interface as a
  compiled control only.
- If the residual bank never improves a non-arithmetic task, remove it from the
  default active path rather than claiming it is useful.
- Do not use the 100% mod-64/mod-32 synthetic score as a language-model claim.
