# taklif8.md — Pretrained-model conversion experiments: execution order and stop gates

## Purpose

`taklif3.md` through `taklif7.md` should not be implemented as one giant change.

This file defines the order, cheap falsification gates, and stop conditions so expensive experiments are only run after cheaper assumptions survive.

## Rule 1 — do not jump to 8B

The sequence is deliberately bottom-up:

```text
architecture correctness
-> sparse FFN conversion
-> hybrid model
-> progressive transplant
-> small-model scaling
-> only then 8B
```

A failed stage blocks dependent stages until the cause is understood.

---

# Stage 0 — finish current Neural Engine architecture validation

Before pretrained conversion work consumes the main research budget:

- confirm Typed Register 100M result on additional seed(s);
- run route-reuse analysis;
- distinguish reusable operator circuits from memorized operand/pair routes;
- begin harder/never-seen composition tests;
- preserve `taklif2.md` as the general reasoning-core roadmap.

Pretrained conversion can be developed in parallel if it does not interrupt these controls.

---

# Stage 1 — taklif3: exact FFN circuit decomposition

Target: smallest practical Qwen3 dense checkpoint.

Cost expectation: mostly engineering/inference; no meaningful pretraining budget should be required.

Required result:

```text
original Qwen
vs
all-circuits converted Qwen

logits/perplexity ~= identical
```

STOP if conversion itself causes material degradation.

Do not train a router until the exact decomposition is correct.

---

# Stage 2 — taklif4: learned FFN sparsification

Start from the Stage-1 converted checkpoint.

Schedule:

```text
100% active
-> 75%
-> 50%
-> 32 circuits
-> 16 circuits
-> 8 circuits
```

At every level record quality/compute curve.

Primary success question:

> Does learned routing beat simple pruning at the same active budget?

Recommended gate before Stage 3:

```text
>= 4x FFN active-compute reduction
with small quality degradation
and learned routing clearly better than naive controls
```

If only 1.2x-1.5x useful sparsity is possible, investigate before deeper architecture transplant.

---

# Stage 3 — taklif5: hybrid Qwen + NE

First hybrid:

```text
all original attention retained
FFN replaced by routed NE circuits
```

Then optionally add persistent working-memory state.

Only after this is stable test reduced attention frequency:

```text
every layer attention
-> every 2nd layer
-> every 4th layer
```

STOP reducing attention when quality/latency tradeoff becomes worse.

A successful hybrid is itself a valid endpoint; full Transformer removal is not mandatory.

---

# Stage 4 — taklif6: progressive attention transplant

Replace one/few attention blocks at a time.

Suggested ladder:

```text
1 block
-> 2-4 blocks
-> 25% of blocks
-> 50%
-> majority
```

Each stage gets:

- local layer matching;
- end-to-end logits distillation;
- language/reasoning evaluation;
- long-context evaluation;
- systems measurements.

STOP when additional replacement causes unrecoverable quality loss.

Keep the best hybrid checkpoint.

---

# Stage 5 — scale the conversion method

Only after the smallest model works:

```text
0.6B-class
-> 1.7B-class
-> optional 4B
-> 8B
```

Do not assume a method that works at 0.6B automatically works at 8B.

At each scale measure:

```text
quality retained
active fraction
GPU-hours of transfer
training tokens used
latency
throughput
RAM/VRAM footprint
```

The key scientific curve is not only quality vs parameters. It is:

```text
quality retained
vs
active compute
vs
transfer cost
```

---

# Stage 6 — 8B proof

8B is justified only if earlier stages show a reproducible favorable curve.

8B should answer a business-relevant question:

> Can a real pretrained 8B-class model keep near-original useful capability while Neural Engine materially reduces active compute/memory/runtime cost?

This is more valuable than training an undertrained 8B from scratch merely to say that an 8B Neural Engine exists.

---

# Required experiment log

Every stage must record:

```text
commit SHA
teacher checkpoint
student checkpoint
seed
training tokens
GPU model
GPU-hours
peak VRAM
system RAM
active params
FLOPs/MAC proxy
latency
throughput
perplexity
benchmark scores
teacher-student KL
failure notes
```

Negative results must be preserved.

---

# Decision tree

```text
Taklif3 exact conversion works?
  NO -> fix decomposition, stop
  YES
   |
Taklif4 useful sparse routing works?
  NO -> router/circuit research, stop deeper transplant
  YES
   |
Taklif5 hybrid works?
  NO -> keep sparse-FFN conversion as endpoint
  YES
   |
Taklif6 attention transplant works?
  NO -> hybrid is endpoint
  YES
   |
Scale 0.6B -> 1.7B -> 4B -> 8B
```

This structure intentionally makes every partial success useful instead of treating full Qwen -> pure Neural Engine conversion as the only acceptable outcome.
