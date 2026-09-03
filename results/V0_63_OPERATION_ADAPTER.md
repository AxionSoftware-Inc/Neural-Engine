# V0.63 Operation-Conditioned Adapter Screen

## Question

V0.62 showed that the prior-free continuous DynamicRegister path reaches
100% on its training compositions but only 77.86% mean on genuine
non-modular held-out operation orders. The suspected problem is that the
continuous state and sparse router can entangle an operation with a specific
composition context. This screen adds a small low-rank adapter shared by
operation type and reused at every recurrent step.

The adapter is not a modular transition table. It has one low-rank down/up
pair and bias per primitive operation, adds its state transform to the query,
and contributes about 38k parameters. The main experiment uses rank 16 and
scale 1.0.

## Results

### Genuine non-modular composition

Both seeds use ordinary integer operations on values `0--3`, no modular
reduction, no modular prior, no Macro-Cells, 9,000 steps, and the same two
held-out operation orders as V0.62.

| Variant | Seed | Total params | Active estimate | Train | Held-out | add -> multiply | multiply -> add |
|---|---:|---:|---:|---:|---:|---:|---:|
| Continuous baseline | 17 | 3,246,787 | 1,445,072 | 100.00% | 72.12% | 73.14% | 71.09% |
| Continuous baseline | 18 | 3,246,787 | 1,445,072 | 100.00% | 83.59% | 81.45% | 85.74% |
| Operation-step route | 17 | 3,246,787 | 1,445,072 | 100.00% | 79.15% | 77.05% | 81.25% |
| Operation-step route | 18 | 3,246,787 | 1,445,072 | 100.00% | 77.15% | 79.39% | 74.90% |
| Operation adapter | 17 | 3,284,803 | 1,483,088 | 100.00% | 84.18% | 78.91% | 89.45% |
| Operation adapter | 18 | 3,284,803 | 1,483,088 | 100.00% | 84.18% | 79.20% | 89.16% |
| **Two-seed mean, baseline** | -- | -- | -- | **100.00%** | **77.86%** | -- | -- |
| **Two-seed mean, operation-step** | -- | -- | -- | **100.00%** | **78.15%** | -- | -- |
| **Two-seed mean, operation adapter** | -- | -- | -- | **100.00%** | **84.18%** | **79.05%** | **89.31%** |

The operation adapter improves the held-out mean by **6.32 percentage
points** over the baseline, with essentially no seed variance in this
screen. Operation-step routing alone is not a reliable improvement: its
mean gain is only **0.29 points** and its seed spread remains large.

### Modular strict gate

These runs use train values `0--31`, unseen eval values `32--63`, train depths
`1--4`, unseen eval depths `5--8`, no Macro-Cells, and random modular
templates.

| Modulus | Variant | Seed 17 | Seed 18 | Two-seed mean |
|---:|---|---:|---:|---:|
| 64 | Templates, no adapter | 96.44% | 96.63% | 96.53% |
| 64 | Templates + operation adapter | 97.05% | 96.75% | 96.90% |
| 32 | Templates, no adapter | 96.04% | 95.02% | 95.53% |
| 32 | Templates + operation adapter | 94.41% | 93.60% | 94.01% |

The adapter gives a small **+0.37-point** mean on mod-64 but a **−1.53-point**
mean on mod-32. It therefore cannot be made the unconditional modular
default from this evidence.

## Interpretation

The shared operation adapter is the strongest current fix for the
prior-free continuous composition failure. Its operation-only parameter
sharing gives the model a reusable primitive lane without exposing the
modular algebra.

The mod-32 regression is equally important. The adapter is not universally
compatible with the modular template state; it can interfere with the
compact exact modular interface, especially as modulus and output dimension
change. The correct current design is conditional: adapter enabled for the
prior-free continuous lane, disabled for the modular-template reference
unless a gated merge proves non-interference.

## Decision

1. Accept the operation adapter as an experimental improvement for the
   prior-free non-modular composition path.
2. Keep `modular_prior: true` reference runs without the adapter for now;
   record the adapter as optional, not as a universal default.
3. Reject operation-step-only routing as a standalone solution because its
   two-seed gain is negligible and unstable.
4. Do not increase capacity to 500M/700M/1B. The largest gain came from a
   38k-parameter interface change, not from a larger bank.
5. Next test a zero-initialized learnable gate around the operation adapter.
   The gate should be able to preserve the modular path at zero contribution
   while opening only when the continuous state benefits from the adapter.

## Run records

- `results/runs/ne_dynamic_300m_composition_nonmodular_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_seed18_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_step_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_step_seed18_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_seed18_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_operation_adapter_strict_seed17_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_operation_adapter_strict_seed18_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_operation_adapter_mod32_strict_seed17_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_operation_adapter_mod32_strict_seed18_9000.json`
