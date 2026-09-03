# V0.62 Genuine Non-Modular Composition

## Question

V0.61 reached 98.71% on held-out operation orders, but that benchmark still
used mod-64 targets. This experiment removes modular reduction from the
generator and checks whether the same attention-free DynamicRegister can
compose integer operations without a modular transition prior.

The task uses values `0--3`, ordinary integer `add`, `subtract`, and
`multiply`, and a label offset of 16 so every two-operation result fits in
the 64-class output head. Training contains seven ordered operation pairs;
evaluation holds out `add -> multiply` and `multiply -> add`. Both runs use
the same 23,600-entry factorized virtual bank, shared factor mix, no
Macro-Cells, and `modular_prior: false`.

## Results

| Variant | Seed | Total params | Active estimate | Train mean | Held-out mean | add -> multiply | multiply -> add |
|---|---:|---:|---:|---:|---:|---:|---:|
| DynamicRegister, no modular reduction | 17 | 3,246,787 | 1,445,072 | 100.00% | 72.12% | 73.14% | 71.09% |
| DynamicRegister, no modular reduction | 18 | 3,246,787 | 1,445,072 | 100.00% | 83.59% | 81.45% | 85.74% |
| **Two-seed mean** | -- | -- | -- | **100.00%** | **77.86%** | **77.29%** | **78.42%** |

The model uses two recurrent register updates, one per operation. There is no
attention, dense transition table, modular template bank, or Macro-Cell bank.
The active estimate is about 1.45M parameters.

## Interpretation

This is a negative result for the current general-purpose composition path.
The model memorizes the seven training compositions but does not reliably
factor the primitive operations into reusable state transitions when modular
algebra is removed. The large seed spread also indicates that the current
continuous state/circuit interface is not yet a stable abstraction for this
task.

The comparison localizes the source of V0.61's high score: the mod-64
environment supplies strong algebraic regularity, and V0.60's modular
template interface exploits it. That is a valuable specialized path, but it
must not be presented as proof of general compositional reasoning.

This result does not prove that the overall architecture cannot solve
non-modular composition. The test is deliberately small and the current
interface has no explicit operation-specific state contract. It does prove
that increasing the virtual circuit bank alone is not the right next move.

## Decision

1. Mark the current prior-free continuous-state route as **not validated**
   for genuine non-modular composition.
2. Keep the modular-template line as a specialized modular-arithmetic result,
   with its task-specific limitation documented.
3. Do not jump to 500M/700M/1B capacity; V0.59 and this test both say the
   bottleneck is representation/interface design, not raw bank size.
4. The next architecture experiment should add an explicit, lightweight
   operation-state interface that is learned rather than hard-coded, then
   rerun both the mod-64 and no-modulus gates with identical budgets.

## Run records

- `results/runs/ne_dynamic_300m_composition_nonmodular_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_seed18_9000.json`
