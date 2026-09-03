# V0.60 Modular Value-State Interface

## Question

The 300M virtual circuit bank did not improve the converged depth-5--8
result over the 20M-class control. This experiment tests the suspected
architectural bottleneck directly: can a small, reusable value-state
interface generalize to values that never appeared during training?

The strict gate trains on values `0--31` and evaluates on unseen values
`32--63`. Training uses operation depths `1--4`; evaluation uses unseen
depths `5--8`, with 1,024 examples per depth and 9,000 optimizer steps.
All variants remain attention-free and use the Dynamic Register path.

## Results

| Variant | Seed | Total params | Active estimate | Depth 5 | Depth 6 | Depth 7 | Depth 8 | Mean eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300M Macro, learned value encoder | 17 | 10,049,747 | 1,458,896 | -- | -- | -- | -- | 37.08% |
| 300M Macro, learned value encoder | 18 | 10,049,747 | 1,458,896 | -- | -- | -- | -- | 36.72% |
| 300M Macro, fixed Fourier encoder | 17 | 10,044,371 | 1,453,520 | 29.20% | 25.78% | 23.14% | 21.78% | 24.98% |
| 300M Macro, random modular templates | 17 | 10,074,844 | 1,483,993 | 99.32% | 99.22% | 95.70% | 89.45% | 95.92% |
| 300M Macro, random modular templates | 18 | 10,074,844 | 1,483,993 | 99.71% | 99.32% | 96.39% | 91.80% | 96.80% |
| 300M, Macro disabled, random modular templates | 17 | 3,285,708 | 1,483,993 | 99.51% | 99.32% | 96.29% | 90.63% | 96.44% |
| 300M, Macro disabled, random modular templates | 18 | 3,285,708 | 1,483,993 | 99.80% | 99.41% | 96.19% | 91.11% | 96.63% |

The two-seed mean for the Macro-enabled template model is **96.36%**. The
Macro-disabled control is **96.53%**, a small **+0.17 percentage-point**
change with about 67% fewer total parameters. This is not evidence that
Macro-Cells are harmful; on this arithmetic gate they are simply redundant
once the modular interface is present.

The learned-only value encoder remains at **36.90%** mean and the fixed
Fourier encoder reaches only **24.98%** in its available 9,000-step seed-17
run. Therefore the failure is not fixed by a generic value representation.
The useful signal comes from the modular state interface.

## Interpretation

The modular-template path creates exact one-hot states for the values and
learns a small operation-template selector over modular add, subtract, and
multiply primitive states. The learned template logits are near-diagonal for
all three operations in both seeds, so the model discovers an operation
aligned with each primitive instead of learning a dense value transition
table. The sparse circuit bank is still available, but it is not carrying
the strict value-generalization result: the Macro-enabled runs used only
93/256 and 109/256 Macro-Cells in the evaluation audit, while the
Macro-disabled control has no Macro bank.

This is a strong localization of the previous capacity problem, not a claim
of general arithmetic intelligence. The current prior exposes the mod-64
algebra used by this synthetic task. It must be tested against a second
modulus and a composition task without the same exact modular prior before
we call the architecture generally scalable.

## Decision

1. Keep the Macro-disabled modular-template configuration as the current
   reference for this gate.
2. Do not spend compute on 500M, 700M, or 1B Macro banks yet; V0.59 already
   showed no meaningful gain from the 300M bank.
3. Next test the same interface under a second modulus and then on a
   non-modular compositional task. A positive result would validate the
   interface; failure would mean the current gain is task-specific wiring.
4. Treat the fixed Fourier and learned-only strict-value variants as rejected
   for this gate, not as production architectures.

## Run records

- `results/runs/ne_dynamic_300m_macro_depth8_strict_values_seed17_9000.json`
- `results/runs/ne_dynamic_300m_macro_depth8_strict_values_seed18_9000.json`
- `results/runs/ne_dynamic_300m_macro_depth8_fixed_fourier_strict_seed17_9000.json`
- `results/runs/ne_dynamic_300m_macro_depth8_modular_templates_strict_seed17_9000.json`
- `results/runs/ne_dynamic_300m_macro_depth8_modular_templates_strict_seed18_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_strict_seed17_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_strict_seed18_9000.json`
