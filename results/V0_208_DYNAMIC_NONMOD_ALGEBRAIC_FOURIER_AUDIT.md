# V0.208 Range-Aware Algebraic/Fourier State Bridge

## Question

V0.206's exact `x,x^2` packet repaired much of the `0--7` depth-4 failure,
but V0.207 showed that the same packet collapsed on operands `0--15`. Was
the remaining bottleneck the bridge from a numeric state to the factorized
digit readout rather than the state value itself?

## Architecture

The fixed algebraic packet still stores normalized `x` and `x^2` and uses the
same exact ordinary integer transitions. The learned bridge now receives the
packet plus multiscale Fourier coordinates for the current `x`: periods
`128`, `16,384`, and `524,288`, each with harmonics `1,2,4,8,16,32,64`.
This adds `42` bounded phase features and changes the learned projection from
`2 -> state_dim` to `44 -> state_dim`. Router, circuit bank, recurrent writer,
and factorized output remain unchanged. The Fourier periods are intentionally
aligned with the base-128 digit interface, so this is a range-aware readout
hypothesis, not a claim of an unstructured universal codec.

## Protocol

The 300M-class factorized DynamicRegister uses ordinary non-modular arithmetic,
operands `0--15`, train depths `1--2`, held-out depths `3--4`, target offset
`65,536`, `524,288` output classes, batch size `512`, and `3,000` CUDA steps.
The comparison is against the matched V0.207 polynomial2 bridge; both use the
same seeds and data protocol. Training uses compact digit logits and avoids
materializing the full Cartesian class matrix; evaluation still reconstructs
exact class logits for accuracy and CE.

## Results

| variant | seed | held-out | depth 3 | depth 4 | train accuracy | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|
| polynomial2 | 17 | 65.63% | 76.17% | 55.08% | 100.00% | 8.83M | 3.53M |
| Fourier bridge | 17 | 93.07% | 95.12% | 91.02% | 100.00% | 8.85M | 3.55M |
| polynomial2 | 18 | 63.28% | 75.20% | 51.37% | 99.90% | 8.83M | 3.53M |
| Fourier bridge | 18 | 95.51% | 97.66% | 93.36% | 100.00% | 8.85M | 3.55M |
| **Fourier mean** | — | **94.29%** | **96.39%** | **92.09%** | **100.00%** | — | — |
| **paired Fourier − polynomial2** | — | **+29.83 pp** | **+20.70 pp** | **+38.96 pp** | — | — | — |

The Fourier bridge adds only `16,128` learned projection parameters. Evaluation
factor-row usage remains broad, so the gain is not explained by opening more
circuits or reducing router starvation.

## Interpretation

This is a large and repeatable quality signal. On the wider `0--15` domain,
the same exact algebraic state that failed at `64.45%` becomes a `94.29%`
held-out solution when the learned bridge receives range-aware phase
coordinates. The dominant bottleneck was therefore the value-to-readout
interface, not raw circuit capacity and not merely training duration.

The result still has two limits. The semantic state transition is fixed, and
the periods are chosen around the base-128 output codec. A wider or shifted
range can expose aliasing. The next test is a `0--31` operand stress with the
same periods and a larger exact output space; no model scale increase is
justified before that gate.

## Decision

**STRONG POSITIVE P-004 SIGNAL; RETAIN AS LEADING OPT-IN ARCHITECTURE.** Do not
make it the default Native Engine or claim universal arithmetic yet. Preserve
the 0--15 checkpoints, test wider-range generalization, and keep the active
parameter accounting separate from the full output materialization cost.

## Artifacts

- `neural_engine/dynamic_register.py` (`algebraic_state_mode: polynomial2_fourier`)
- `train_dynamic_composition.py` (compact factorized training path)
- `tests/test_dynamic_register.py`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_15_factorized_algebraic_fourier.yaml`
- `results/runs/nonmod_depth4_values0_15_factorized_algebraic_fourier_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_15_factorized_algebraic_fourier_seed18_3000.json`
