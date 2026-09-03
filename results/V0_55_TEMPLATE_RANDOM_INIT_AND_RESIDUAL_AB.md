# V0.55 template random-init and sparse-residual A/B

Status: **modular template interface promoted for the synthetic line; sparse
residual remains optional and conditional**

## Why this follow-up matters

The V0.54 identity-initialized template screen reached 100%, but identity
initialization is already close to the declared operation mapping. This report
checks random initialization, a second modulus, and whether the large sparse
factorized bank is actually needed for the arithmetic gate.

## Random-init template results

The standalone trainable template register uses no dense transition table. It
learns the mapping from operation token to three fixed equivariant primitives.
The mapping starts from a small random matrix, not the identity.

| Modulus | Seed | Steps | Train | Unseen value/depth | Params |
|---:|---:|---:|---:|---:|---:|
| 64 | 17 | 3,000 | 99.07% | 64.16% | 4,169 |
| 64 | 18 | 3,000 | 99.37% | 69.73% | 4,169 |
| 64 | 17 | 10,000 | 100.00% | 100.00% | 4,169 |
| 64 | 18 | 10,000 | 100.00% | 100.00% | 4,169 |
| 32 | 17 | 10,000 | 100.00% | 100.00% | 1,065 |
| 32 | 18 | 10,000 | 100.00% | 100.00% | 1,065 |

The 10k runs learn a near-diagonal token-to-primitive matrix in both moduli.
The mod-32 result is important: the effect is not tied to the 64-class label
space. The learned model still receives modular-equivariant wiring by design,
so this validates the interface family, not unconstrained arithmetic discovery.

## Dynamic Register residual A/B

Both variants use the 20M shared-factor Dynamic Register, random template
initialization, and the strict gate: train values 0–31/depths 1–4, evaluate
values 32–63/depths 5–6.

| Variant | Steps | Train | OOD | Active estimate | Time |
|---|---:|---:|---:|---:|---:|
| Template prior, no circuit residual | 3,000 | 99.85% | 90.72% | 1.479M | 117s |
| Template prior + sparse residual | 3,000 | 99.90% | 91.21% | 1.479M | 355s |
| Template prior, no circuit residual | 10,000 | 100.00% | 99.71% | 1.479M | 391s |
| Template prior + sparse residual | 10,000 | 100.00% | 99.51% | 1.479M | 1,161s |

At equal 10k training budget the no-residual path is slightly better and much
faster. The residual does not supply the modular arithmetic; the compact
template register does. Keep the sparse bank available for future learned
behaviors, but do not count it as necessary arithmetic capacity or force it
active on this task.

## Decision

Promote the random-init modular template interface as the current synthetic
architecture direction. Do not scale the 20M sparse bank to 300M/1B merely to
solve this OOD gate; the structured interface solves it at tiny scale.

Before claiming a general Neural Engine architecture, run:

1. another modulus and renamed operation tokens with no benchmark-specific
   identity initialization;
2. a non-arithmetic compositional task where the residual bank can contribute;
3. a fair no-residual versus residual latency/memory audit; and
4. learned equivariant templates whose primitive wiring is not simply copied
   from the target rule.

## Reproduction run IDs

```text
ne_modular_templates_random_unseen_values_seed17_3000
ne_modular_templates_random_unseen_values_seed18_3000
ne_modular_templates_random_unseen_values_seed17_10000
ne_modular_templates_random_unseen_values_seed18_10000
ne_modular_templates_mod32_random_seed17_10000
ne_modular_templates_mod32_random_seed18_10000
ne_dynamic_20m_modular_templates_no_residual_unseen_values_3000
ne_dynamic_20m_modular_templates_random_residual_unseen_values_3000
ne_dynamic_20m_modular_templates_no_residual_unseen_values_10000
ne_dynamic_20m_modular_templates_random_residual_unseen_values_10000
```
