# V0.59 300M Macro-Cell Long-Depth Screen

## Purpose

After the 256-cell model passed the converged depth-5--8 gate, this screen
tests whether the larger 300M-class virtual circuit bank improves quality. The
model keeps the attention-free recurrent register, factorized sparse circuit
routing, shared factor mix, and a 256-cell Macro-Cell bank. It trains on
depths 1--4 and evaluates on unseen depths 5--8.

## Results

Both seeds use 9,000 optimizer steps and 1,024 evaluation examples per depth.

| Model | Seed | Total params | Active estimate | Depth 5 | Depth 6 | Depth 7 | Depth 8 | Mean eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20M Macro, per-address mix | 17 | 8,538,065 | 1,458,912 | 99.61% | 99.51% | 99.41% | 99.22% | 99.44% |
| 20M Macro, per-address mix | 18 | 8,538,065 | 1,458,912 | 99.41% | 99.90% | 99.02% | 99.41% | 99.44% |
| 300M Macro, shared mix | 17 | 10,049,747 | 1,458,896 | 99.71% | 99.80% | 99.02% | 99.02% | 99.39% |
| 300M Macro, shared mix | 18 | 10,049,747 | 1,458,896 | 99.80% | 99.90% | 99.41% | 99.41% | 99.63% |

The two-seed means are **99.44%** for the 20M-class control and **99.51%**
for the 300M-class model: a negligible **+0.07 percentage-point** change.
Both 300M runs route through all 256 Macro-Cells. The larger virtual micro
bank reaches only about 10--12% of its virtual circuits in the audit, while
all 154 shared factor rows are exercised.

## Decision

Do not jump to 500M, 700M, or 1B based on this gate. The active computation is
almost unchanged and the larger virtual bank does not produce a meaningful
quality gain once both models converge. The next acceptance test is strict
value generalization: train on values 0--31 and evaluate on values 32--63.
That test determines whether the current architecture has learned a reusable
operation rather than memorized the observed value range.

Run records:

- `results/runs/ne_dynamic_300m_macro_depth8_seed17_9000.json`
- `results/runs/ne_dynamic_300m_macro_depth8_seed18_9000.json`

