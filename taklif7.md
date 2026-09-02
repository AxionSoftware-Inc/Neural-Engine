# Taklif 7 — trainable modular-equivariant circuit prior

Status: **trainable-template screen positive on two seeds; second-modulus and
sparse-residual validation still required**

## Evidence

The learned Dynamic Register Machine reaches only about 35% on values 32–63
when trained on values 0–31, even after capacity and routing ablations. A
fixed mod-64 transition register reaches 100% without attention, and adding
that prior to the sparse recurrent model also reaches 100%.

This isolates the missing ingredient as modular composition structure. It does
not justify claiming that a fixed arithmetic table is a generally learned
neural network.

## Proposed architecture

```text
raw value / operation tokens
          |
  compact phase or one-hot register
          |
  trainable equivariant operator template
          |
 sparse factorized residual circuits + recurrent writer
          |
       class readout
```

The transition should be parameterized so that add/subtract use circular
translation and multiplication uses a reusable modular permutation/action,
instead of allocating one unrelated parameter row for every value. The sparse
factorized bank remains the adaptive residual path; the structural part must
not become a dense 64×64×64 table.

The first screen now implements this as `TrainableModularTemplateRegister`.
With only 4,169 parameters it reaches 100% on both seeds of the strict
unseen-value/depth gate. This is a positive signal, but the mod-64 wiring is
still supplied by the architecture, so it is not yet a generalization claim.

## Required A/B tests

1. fixed-prior control;
2. trainable equivariant templates with random initialization;
3. the same templates initialized from the fixed-prior solution;
4. learned-only Dynamic Register reference;
5. values 0–31 → 32–63 OOD, unseen depths 5–6, two seeds;
6. a second modulus or renamed primitive set to check that the result is not
   merely memorization of mod-64 labels; and
7. active trainable parameters, route reuse, latency, and memory.

## Acceptance gate

The trainable template must beat the 34–35% learned-only OOD result by a large
margin on two seeds without using a target oracle or a dense transition table.
100% is the structural ceiling, not the minimum requirement. The proposal is
promoted only if it also transfers to a second algebraic setting.

## Rejected shortcuts

- increasing 300M → 500M → 700M without changing the representation;
- forcing more active circuits for difficult examples;
- using Qwen/Transformer logits as a teacher for this synthetic arithmetic
  gate; and
- presenting the exact modular control as learned generalization.
