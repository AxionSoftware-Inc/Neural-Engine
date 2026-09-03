# V0.65 No-Residual Circuit Control

## Question

V0.63's operation adapter improves prior-free composition, but it was unclear
whether the sparse circuit residual was helping or merely memorizing training
compositions. This control keeps the same operation adapter and routing path
but sets `circuit_residual_scale: 0.0`, so selected circuits cannot contribute
a state delta.

## Results

Both seeds use ordinary integer operations on values `0--3`, no modular
reduction, no modular prior, no Macro-Cells, 9,000 steps, and held-out
`add -> multiply` / `multiply -> add` orders.

| Variant | Seed | Total params | Active estimate | Train | Held-out |
|---|---:|---:|---:|---:|---:|
| Full residual + fixed operation adapter | 17 | 3,284,803 | 1,483,088 | 100.00% | 84.18% |
| Full residual + fixed operation adapter | 18 | 3,284,803 | 1,483,088 | 100.00% | 84.18% |
| No circuit residual + operation adapter | 17 | 3,284,803 | 1,483,088 | 100.00% | 79.59% |
| No circuit residual + operation adapter | 18 | 3,284,803 | 1,483,088 | 100.00% | 78.81% |
| **Two-seed mean, full residual** | -- | -- | -- | **100.00%** | **84.18%** |
| **Two-seed mean, no residual** | -- | -- | -- | **100.00%** | **79.20%** |

The no-residual control keeps routing computation and the same active-parameter
estimate, but its selected circuit outputs are replaced by zeros. It loses
**4.98 percentage points** on held-out composition.

## Interpretation

The sparse circuit residual is not only a source of memorization. It carries
useful computation for the non-modular composition task. Removing it leaves
the operation adapter and recurrent writer, but that interface alone is not
enough to reach the full result.

The remaining problem is therefore an interaction problem: the circuit bank
must contribute useful reusable computation while avoiding composition-specific
entanglement. The right next work is circuit/interface factorization and
route analysis, not deleting the circuit bank or increasing its virtual size.

## Decision

1. Reject the no-residual path as the continuous reference.
2. Keep full circuit residual plus fixed operation adapter for prior-free
   composition experiments.
3. Keep the adapter-free modular-template path as the modular reference;
   the gated adapter did not fully protect it across moduli.
4. Do not increase capacity to 500M/700M/1B yet. The no-residual control
   confirms the bottleneck is computation/interface quality, not bank size.

## Run records

- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_no_residual_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_no_residual_seed18_9000.json`
