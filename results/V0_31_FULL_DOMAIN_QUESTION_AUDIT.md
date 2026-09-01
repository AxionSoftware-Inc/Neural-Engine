# V0.31 full-domain question and benchmark audit

This audit is inference-only. No checkpoint was trained or fine-tuned during
this run. The goal was to determine whether the apparent capacity effect came
from the question system, the restricted operand range, or the model/router.

## Protocol

Every checkpoint received the same fixed-format composition questions:

```text
(a, b, op1) -> partial -> (partial, c, op2) -> target mod 64
```

The deterministic grid was expanded from `16^3` to the complete operand domain
`64^3 = 262,144` examples per ordered operator pair. The two difficult pairs
were evaluated consistently as `add_then_multiply` and `multiply_then_add`.
The all-pairs controls used all nine ordered pairs. The evaluator now accepts
repeated `--pair` arguments so a checkpoint's config cannot silently change the
question set:

```powershell
python evaluate_composition.py `
  --checkpoint results/checkpoints/ne_typed_register_100m_all_10000.pt `
  --grid-size 64 --pair add_then_multiply --pair multiply_then_add `
  --batch-size 512 --device cuda
```

Repeated identical inputs were deterministic in evaluation mode: the 100M
checkpoint returned the same prediction for all repeated copies, with maximum
logit difference below `2e-6`.

## Full-domain results

| Checkpoint | Stored params | Same two hard pairs | All nine pairs |
|---|---:|---:|---:|
| 20M all-pairs | 19.81M | 60.62% | 58.10% |
| 50M all-pairs | 51.04M | 60.67% | 58.10% |
| 100M all-pairs | 103.26M | 62.41% | **59.23%** |
| 300M all-pairs | 309.55M | 61.40% | 58.60% |
| 20M hidden-stage | 19.81M | **64.89%** | — |
| 50M hidden-stage | 51.04M | 63.06% | — |
| 100M hidden-stage | 103.26M | **89.16%** | — |
| 300M hidden-stage | 309.55M | 85.42% | — |

The hidden-stage rows are evaluated only on the two hidden pairs because their
checkpoint configs select those pairs. They are not strict zero-shot rows:
the preceding all-pairs phase exposed all nine operator combinations, after
which the 5,000-step stage adapted on the six visible pairs.

For the hidden-stage full-domain rows, per-pair accuracy was:

| Checkpoint | add → multiply | multiply → add |
|---|---:|---:|
| 20M | 70.44% | 59.35% |
| 50M | 69.49% | 56.63% |
| 100M | **90.76%** | **87.57%** |
| 300M | 85.35% | 85.48% |

The old `16^3` hidden scores were 65.56%, 63.34%, 89.87%, and 89.20% for
20M/50M/100M/300M. Moving to the complete `64^3` domain changes them by
−0.67, −0.28, −0.71, and −3.78 points respectively. Therefore the restricted
operand range is not the main explanation for the 100M result, although it
does matter for 300M.

Ten hand-selected edge/interior questions were also sent to every hidden-stage
checkpoint, including values `0`, `1`, `31`, `32`, and `63`. The 100M and 300M
models answered all ten correctly. The 20M and 50M models made the same one
mistake: `(63 * 63 + 63) mod 64 = 0` was predicted as `32`.

## Diagnosis

There is no evidence that the evaluator is asking different questions within
one run or that inference is nondeterministic. The important benchmark issues
are instead:

1. `16^3` is a partial value domain, so the primary reference should be the
   full `64^3` grid.
2. The former hidden score measured post-exposure adaptation, not strict
   unseen-composition generalization.
3. All-pairs and hidden-stage commands implicitly evaluated different pair
   sets. The explicit `--pair` override removes that ambiguity.

The current result supports this hypothesis: 20M is already large enough to
represent this narrow two-hop arithmetic task, and the active path stays near
1.51M parameters. The quality bottleneck is route/circuit credit assignment,
not raw stored capacity. Around 100M, one seed found a much better active
solution; at 300M, extra stored rows did not improve the active computation and
likely increased route fragmentation/optimization variance. This is why the
scaling curve is non-monotonic: 100M wins, 300M falls back.

## Decision

Do not jump to 700M/1B based on this benchmark. The next meaningful experiment
is a corrected strict-zero-shot protocol, with the six visible pairs excluded
from training from step 1, the two hidden pairs held out completely, full
`64^3` evaluation, explicit pair selection, and at least two seeds at 20M and
100M. Architecture work should target credit assignment and route reuse before
adding more dormant circuit rows.
