# V0.196 Non-Modular Depth-4 Typed-Write Validation

## Question

V0.195's 3,000-step screen reached `85.16%` on unseen non-modular depths
3--4. This validation asks whether the signal survives the longer 9,000-step
training budget used by the established DynamicRegister references.

## Protocol

The configuration is unchanged from V0.195: 300M-class factorized virtual
bank, 23,600 virtual circuits, 154 factor rows, operation-specific circuit
banks, rank-16 operation query adapter, rank-16 typed-write adapter, values
0--3, ordinary integer add/subtract/multiply, target offset 64, train depths
1--2 and held-out depths 3--4. Only the training budget and evaluation sample
count were increased: 9,000 steps and 256 examples per held-out depth.

## Results

| seed | train depth 1--2 | held-out depth 3--4 | depth 3 | depth 4 |
|---:|---:|---:|---:|---:|
| 17 | 100.00% | 86.33% | 92.19% | 80.47% |
| 18 | 100.00% | 85.55% | 91.80% | 79.30% |
| **mean** | **100.00%** | **85.94%** | **91.99%** | **79.88%** |

The 9k mean is `+0.78 pp` above the V0.195 3k mean, and both seeds improve
over their respective 3k screens (`+3.13 pp` and `−1.56 pp`). The remaining
depth-4 gap is consistent across seeds. No NaN or route collapse occurred;
the 154 factor rows were used broadly on held-out depth-3/4 evaluation.

## Decision

**VALIDATED AS A POSITIVE ARCHITECTURE SIGNAL, NOT YET A SCALE GO.** The
typed-write DynamicRegister can learn depths 1--2 and transfer to unseen
depths 3--4 in ordinary non-modular arithmetic at 9k, so the 3k signal was
not only a short-run artifact. However, `85.94%` is not near-perfect and
there is no comparison here against a larger bank. Do not jump to 500M/700M/1B
yet.

The next inexpensive gate is a third seed on this exact protocol. If it stays
near the two-seed mean, test the same 300M interface on a larger non-modular
operand range or one additional unseen depth. Only after that should capacity
growth be reconsidered, with active-path cost reported separately.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter.yaml`
- `results/runs/nonmod_depth4_typed_write_seed17_9000.json`
- `results/runs/nonmod_depth4_typed_write_seed18_9000.json`
- `results/V0_195_DYNAMIC_NONMOD_DEPTH4_TYPED_WRITE_SCREEN.md`
