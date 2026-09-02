# V0.49 dynamic Neural Register Machine

Status: **main-path positive research signal; scaling law and systems savings
remain unproven**.

## Why this is the main path

The Qwen/attention experiments were useful diagnostics, but they retained the
Transformer and did not approach the intended 4x active-compute target. V0.49
returns to the independent architecture:

- no self-attention;
- no Transformer block;
- no pretrained Qwen weights;
- recurrent typed register/accumulator state;
- hierarchical/factorized sparse circuit routing;
- input-dependent number of executed program steps.

The previous typed-register model hard-coded exactly three stages for a
two-operation arithmetic task. That made it difficult to tell whether larger
capacity learned a reusable computation rule or merely added dormant route
addresses. V0.49 replaces that fixed graph with a left-to-right dynamic
register machine.

```text
program + operands
        |
        v
accumulator register + next operand + operation code
        |
        v
pair/product composition + step code
        |
        v
factorized hierarchical router -> 8 active circuits
        |
        v
register write -> next operation
```

The router does not score the entire virtual bank. The model scans a program
with `1..6` operations, and padded operations are skipped. This is a genuine
attention-free recurrent computation path rather than a Transformer with a
different FFN.

## Benchmark

`DynamicCompositionGenerator` encodes a program as operation tokens followed
by operand tokens. Operations are modular `add`, `subtract`, and `multiply`
over values `0..63`. The target is the left fold of the operation sequence.

The primary falsification split trains only on depths 1–4 and evaluates only
on unseen depths 5–6. Every size uses the same split, seed, batch size, number
of steps, and active circuit count:

```text
train depths: 1..4
eval depths:  5..6
values:       0..63
steps:        3000
batch:        512
active:       8 factorized circuits per executed step
seed:         17 unless noted
```

The reported 20M/100M/300M labels are **virtual circuit-capacity tiers**, not
physical parameter counts. Factorization intentionally keeps physical storage
sublinear; the distinction must not be hidden.

## Results

| Virtual tier | Physical params | Active estimate | Train 1–4 | Held-out 5–6 | Depth 5 | Depth 6 |
|---|---:|---:|---:|---:|---:|---:|
| 20M, seed 17 | 1.74M | 1.45M | 84.67% | 72.85% | 73.44% | 72.27% |
| 100M, seed 17 | 2.42M | 1.45M | 84.18% | 73.83% | 76.56% | 71.09% |
| 300M, seed 17 | 3.30M | 1.45M | 90.43% | **80.86%** | 80.86% | 80.86% |
| 300M, seed 18 | 3.30M | 1.45M | 86.72% | **76.37%** | 75.78% | 76.95% |

The 300M held-out mean across the two seeds is **78.62%**, versus the 20M
seed-17 control at 72.85%. The gain is encouraging because the model never
saw depth 5 or 6 during training. The 300M seed spread is 4.49 percentage
points, so the result is a direction signal, not yet a precise scaling law.

As an all-depth sanity control, the 20M tier trained on depths 1–6 for 5,000
steps reaches 89.45% balanced accuracy. Its average execution is 3.5 of the
6 possible recurrent steps (58.33% active-step fraction). In the primary
holdout, the average is 5.5 of 6 steps because the evaluation intentionally
contains only depths 5–6.

## What the result proves and does not prove

The result supports three claims:

1. a non-attention recurrent register machine can learn a reusable modular
   operation path;
2. unseen longer programs are substantially above chance and improve with
   virtual circuit capacity in this control; and
3. the maximum active circuit count stays fixed at 8 per executed step while
   virtual addresses increase.

It does **not** yet prove:

- general language or reasoning ability;
- true 20M/100M/300M physical stored parameters—the tested banks are
  factorized and physically smaller;
- actual GPU speedup—the current serial PyTorch implementation still has
  dispatch overhead;
- reliable 4x total-model active-parameter reduction;
- zero-shot generalization to unseen numeric values or operation families.

## Decision

Keep Dynamic Register Machine as the new main research path. Freeze the Qwen
transfer line as an auxiliary/negative-result branch and do not spend the next
budget on 700M/1B Qwen conversion.

Do not yet claim the dynamic model is production-ready. The current gate is
**positive but conditional** until the following controls pass.

## Next controls

1. repeat 20M and 100M with seed 18 and use a larger held-out evaluation set;
2. add depth 7–8 and a separate unseen-value-range split;
3. run route-reuse, dead-circuit, and per-depth active-parameter audits;
4. compare factorized and true physical-capacity banks at matched training
   budgets; and
5. replace serial per-sample dispatch with a grouped circuit kernel before
   making any latency claim.

The key acceptance curve is now:

```text
held-out program quality
vs
virtual/physical capacity
vs
active circuits and executed steps
```

If the curve survives those controls, parent-growth to 500M/700M is justified
inside the attention-free Neural Engine. If it fails, the next change should
target register write/composition mathematics—not another pretrained-model
transfer experiment.

Reproduction entry points:

```text
python train_dynamic_composition.py --config configs/ne_dynamic_20m.yaml --steps 3000 --batch-size 512 --device cuda --heldout-depths
python train_dynamic_composition.py --config configs/ne_dynamic_100m.yaml --steps 3000 --batch-size 512 --device cuda --heldout-depths
python train_dynamic_composition.py --config configs/ne_dynamic_300m.yaml --steps 3000 --batch-size 512 --device cuda --heldout-depths --seed 18
```
