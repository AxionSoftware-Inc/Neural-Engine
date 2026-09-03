# V0.61 Prior-Free Composition Holdout

## Question

V0.60 showed that the modular value-state/template interface transfers from
mod-64 to mod-32. This experiment asks a different question: can the new
attention-free register/circuit architecture compose operations it has not
seen together without receiving the exact modular transition prior?

The training split contains seven ordered operation pairs. Evaluation holds
out `add -> multiply` and `multiply -> add`. The input contains primitive
operation tokens and three values; it does not contain a token naming the
final composition. Both runs use the 300M-class virtual bank, shared factor
mix, no Macro-Cells, a learned value encoder, and `modular_prior: false`.

## Results

| Variant | Seed | Total params | Active estimate | Train mean | Held-out mean | add -> multiply | multiply -> add |
|---|---:|---:|---:|---:|---:|---:|---:|
| DynamicRegister, prior-free | 17 | 3,246,787 | 1,445,072 | 99.89% | 99.02% | 98.44% | 99.61% |
| DynamicRegister, prior-free | 18 | 3,246,787 | 1,445,072 | 99.99% | 98.39% | 97.66% | 99.12% |
| **Two-seed mean** | -- | -- | -- | **99.94%** | **98.71%** | **98.05%** | **99.37%** |

The model uses exactly two recurrent register updates for this two-operation
program. No attention, dense transition table, modular template bank, or
Macro-Cell bank is active. The virtual circuit bank remains large, but the
reported active estimate is about 1.45M parameters.

## Interpretation

This is a strong positive architectural signal. The model learns reusable
primitive operation behavior and transfers it to operation orders that were
never present in the training split. The result is not explained by the
V0.60 modular-template path because that path is disabled.

The benchmark still defines the primitive operations modulo 64, so this is
not yet a fully non-modular task. It proves prior-free compositional transfer
within the modular arithmetic environment, not general reasoning over an
arbitrary domain. A genuine non-modular compositional dataset is the next
validation required before making a broader claim.

## Decision

1. Keep this prior-free DynamicRegister configuration as the composition
   reference.
2. Do not increase the virtual bank to 500M/700M/1B for this gate; quality is
   already high and V0.59 found no meaningful capacity gain.
3. Add a genuine non-modular compositional task with the same fixed-layout
   interface and test whether the result survives removal of modular
   arithmetic from the target function.
4. Preserve the V0.60 modular-template line as a specialized fast path for
   modular arithmetic, but do not confuse it with the general architecture.

## Run records

- `results/runs/ne_dynamic_300m_composition_holdout_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_holdout_seed18_9000.json`
