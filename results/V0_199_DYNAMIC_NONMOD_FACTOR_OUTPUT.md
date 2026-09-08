# V0.199 Factorized Output on Non-Modular Values 0--7

## Question

Was the quality drop in the wider non-modular `0--7` task mainly caused by the
32,768-class flat output head, whose parameters dominate the active path?

## Protocol

The experiment keeps the V0.198 300M-class DynamicRegister configuration,
typed-write state interface, factorized circuit bank, train depths `1--2`,
held-out depths `3--4`, ordinary integer arithmetic (`modulus: null`), target
offset `4096`, batch size `512`, and `3,000` CUDA steps. The only architectural
change is the output interface: the target class is decomposed into a high
digit (`256` classes) and low digit (`128` classes), with additive reconstruction
of the exact `32,768`-class logits. Training uses the two digit cross-entropies;
evaluation still uses exact reconstructed-class argmax and full-class loss.

The comparison is against the V0.198 flat-output runs with the same seeds and
protocol. Both variants fit the seen depths to `100%`.

## Results

| variant | seed | held-out depth 3--4 | depth 3 | depth 4 | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|
| flat output | 17 | 72.27% | 81.25% | 63.28% | 19.82M | 14.52M |
| flat output | 18 | 77.34% | 85.16% | 69.53% | 19.82M | 14.52M |
| **flat mean** | — | **74.80%** | **83.20%** | **66.41%** | — | — |
| factorized digits | 17 | 71.09% | 82.81% | 59.38% | 7.35M | 2.05M |
| factorized digits | 18 | 72.85% | 85.16% | 60.55% | 7.35M | 2.05M |
| **factorized mean** | — | **71.97%** | **83.98%** | **59.96%** | — | — |

Relative to the flat head, factorization changes held-out accuracy by
`-2.83 percentage points`, improves depth-3 accuracy by `+0.78 pp`, and lowers
depth-4 accuracy by `-6.45 pp`. Relative to the earlier `0--3` three-seed
reference (`84.70%`), the factorized `0--7` mean is `-12.73 pp`.

## Interpretation and decision

**Rejected as a quality solution; retained as a diagnostic/sparsity option.**
The large flat classifier is a real sparsity problem: factorization reduces
the estimated active path by `12.47M` parameters. However, replacing it with
two small digit heads does not recover the lost depth-4 generalization. The
dominant failure is therefore not only the output-head parameter count. The
typed-write recurrent state/interface and its depth-transfer behavior remain
the leading suspects.

This result does not prove that the circuit bank is too small, and it does not
justify increasing capacity. It also does not prove an inference latency win:
the current implementation reconstructs the full `32,768` logits for exact
class scoring. The gain measured here is parameter/active-path accounting;
an index-only inference kernel would be a separate implementation task.

## Next action

Keep the flat `0--7` run as the quality baseline and the factorized head as an
optional sparse-output branch. Focus the next experiment on the recurrent
state/interface (whether information from earlier operations is preserved and
read correctly at depth 4), with the output head held fixed. Do not increase
the circuit bank until a state-preservation probe shows that the depth-4 signal
reaches the final output.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized.yaml`
- `results/runs/nonmod_depth4_values0_7_factorized_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_7_factorized_seed18_3000.json`
- `neural_engine/dynamic_register.py` (`FactorizedDigitOutput`)
- `train_dynamic_composition.py` (digit loss and target-range validation)
- `tests/test_dynamic_register.py`
