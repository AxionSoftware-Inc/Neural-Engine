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

The first second-seed controls are now complete. The 20M tier scores 75.49%
with seed 18 (two-seed mean **74.17%**), while 100M scores 72.17% (two-seed
mean **73.00%**). Thus the 300M mean remains above both smaller tiers, but the
20M/100M ordering is not monotonic and should not be over-interpreted.

The deeper holdout keeps the same training budget but evaluates unseen depths
5–8 after training only on depths 1–4:

| Virtual tier | Physical params | Held-out 5–8 | Depth 5 | Depth 6 | Depth 7 | Depth 8 |
|---|---:|---:|---:|---:|---:|---:|
| 20M, seed 17 | 1.75M | 72.12% | 78.13% | 70.51% | 71.29% | 68.55% |
| 300M, seed 17 | 3.31M | **75.68%** | 81.64% | 73.44% | 74.41% | 73.24% |

This is a smaller but consistent +3.56 percentage-point capacity signal on
the harder 7–8 depth extension. It is still a single seed and a synthetic
task, so it is evidence for continuation rather than a final scaling claim.

A disjoint value-range control exposes the current weakness. Training uses
only values 0–31 and evaluation uses only 32–63, while depths remain 1–4 to
5–6:

| Virtual tier | Train accuracy | Unseen-value accuracy |
|---|---:|---:|
| 20M, seed 17 | 99.22% | 34.77% |
| 300M, seed 17 | 99.76% | 34.38% |

Increasing virtual capacity does not improve this result. The model is above
the 1.56% random baseline, but it is not yet learning a value-independent
algebraic circuit. This points to the value representation/register-write
path as the next bottleneck, not to insufficient circuit-bank capacity.

## Route audit

The 300M depth-8 run was repeated with route instrumentation and reproduced
the same **75.68%** held-out accuracy. On the 2,048-example evaluation batch,
the model made 106,496 active virtual-circuit selections. These selections
covered 1,777 of 23,600 virtual addresses (**7.53%**), or about 60 selections
per observed virtual address on average. The most frequent address accounted
for only 3.74% of traffic, so the result is route reuse rather than collapse
to one circuit.

Because the bank is factorized, the virtual-address coverage is not the whole
story: 137 of 154 factor rows (**88.96%**) were touched by the same batch.
The remaining 22,000+ virtual addresses were not selected in this finite
audit batch; they should be called *unobserved in the batch*, not permanently
dead, until traffic is measured over a much larger and more varied corpus.
Per-depth virtual-address coverage was 1,173 / 1,226 / 1,297 / 1,334 for
depths 5 / 6 / 7 / 8. This confirms meaningful route reuse and depth-specific
traffic, but does not yet establish that every capacity tier is receiving
useful training gradients.

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
   virtual addresses increase; and
4. the positive capacity signal survives the first 7–8 depth extension.

It does **not** yet prove:

- general language or reasoning ability;
- true 20M/100M/300M physical stored parameters—the tested banks are
  factorized and physically smaller;
- actual GPU speedup—the current serial PyTorch implementation still has
  dispatch overhead;
- reliable 4x total-model active-parameter reduction;
- zero-shot generalization to unseen numeric values or operation families;
- a monotonic scaling law across all virtual tiers and random seeds.

## Decision

Keep Dynamic Register Machine as the new main research path. Freeze the Qwen
transfer line as an auxiliary/negative-result branch and do not spend the next
budget on 700M/1B Qwen conversion.

Do not yet claim the dynamic model is production-ready. The current gate is
**positive but conditional** until the following controls pass.

## Next controls

1. repeat the route audit over a larger corpus and report observed versus
   permanently unused traffic separately;
2. compare factorized and true physical-capacity banks at matched training
   budgets; and
3. replace serial per-sample dispatch with a grouped circuit kernel before
   making any latency claim.

The unseen-value failure must be addressed before treating 500M/700M as a
quality solution. Parent-growth may still be useful for the longer-program
capacity curve, but it is not expected by itself to repair value
representation generalization.

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
python train_dynamic_composition.py --config configs/ne_dynamic_20m_depth8.yaml --steps 3000 --batch-size 512 --device cuda --heldout-depths
python train_dynamic_composition.py --config configs/ne_dynamic_300m_depth8.yaml --steps 3000 --batch-size 512 --device cuda --heldout-depths
python train_dynamic_composition.py --config configs/ne_dynamic_300m.yaml --steps 3000 --batch-size 512 --device cuda --heldout-depths --train-value-max 31 --eval-value-min 32 --eval-value-max 63
```
