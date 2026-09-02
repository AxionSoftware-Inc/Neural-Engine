# V0.50 dynamic generalization diagnostic

Status: **diagnostic complete; current architecture bottleneck isolated**

This control was run before opening the proposal ablations. The 300M dynamic
register model was trained for 3,000 steps on depths 1–4 and values 0–31.
The checkpoint was then evaluated without further training across short and
long programs, both value ranges, and each primitive operation separately.

## Main result

| Evaluation condition | Accuracy |
|---|---:|
| Train range 0–31, depths 1–4 | **99.68%** |
| Train range 0–31, unseen depths 5–6 | **98.44%** |
| Unseen values 32–63, depths 1–4 | 38.72% |
| Unseen values 32–63, depths 5–6 | 33.84% |

The register/composition path generalizes strongly to longer programs inside
the observed value domain. The failure is specifically value extrapolation,
not simply depth, capacity, or inability to execute a long recurrent chain.

## Operation breakdown on unseen values

| Operation | Depths 1–4 | Depths 5–6 |
|---|---:|---:|
| add | 34.18% | 26.17% |
| subtract | 33.11% | 24.22% |
| multiply | **63.55%** | **88.53%** |

The arithmetic asymmetry is the key clue. Multiplication transfers much
better than addition/subtraction, while all three operations are near-perfect
on the train range. This suggests a representation/operation interaction for
modular wraparound rather than a generic lack of model capacity.

As a control, the 300M model trained on the full 0–63 range scores well on
32–63, confirming that the value encoder can represent those tokens. The
problem is learning a rule that extrapolates beyond the values seen during
training.

## Decision gate

Do not interpret the 34% unseen-value score as evidence that 500M/700M will
solve the problem. The 20M and 300M OOD runs were 34.77% and 34.38%, so raw
virtual capacity did not help. The first proposal screen (state width,
reinjection, gated write, circuit rank, parallel mix, and route exploration)
also produced no reliable large improvement. The full screen is recorded in
`V0_51_DYNAMIC_PROPOSALS_09_15.md`.

The diagnostic script is:

```text
python diagnose_dynamic_generalization.py --checkpoint results/checkpoints/ne_dynamic_300m_unseen_values_checkpoint_3000.pt --output results/runs/ne_dynamic_300m_value_generalization_diagnostic.json --examples-per-depth 1024
```

The checkpoint and JSON run artifact are local ignored artifacts; this report
and the script are the reproducible source-controlled record.
