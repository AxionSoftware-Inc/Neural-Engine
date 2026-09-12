# taklif24.md — Elastic Temporary Expansion: task-conditioned grow/shrink reasoning graph

Status: **independent high-risk research proposal; test temporary reasoning expansion only**

> **SCOPE:** Test whether a fixed stored model can temporarily create a larger computation graph for hard inputs and then discard it. This proposal does not add persistent knowledge, continual learning, or long-term parameter growth.

## Hypothesis

A model with stored parameters `theta` can improve hard-task quality by generating temporary modules/circuits `phi(x)` whose size depends on task difficulty:

`y = F(x; theta, phi_1(x), ..., phi_N(x))`

with `N` chosen dynamically. Easy inputs should use a small graph; hard inputs may expand to `2x/4x/8x/16x` temporary logical compute. After the answer, temporary modules are deleted.

The claim is **not** that 27B stored parameters magically become an independently trained 200B model. The claim is that the same stored knowledge may support substantially more reasoning computation when the task requires it.

## V0 experiment

Use a small base model/substrate, not a billion-scale run. Compare fixed-compute control against dynamic budgets `1x, 2x, 4x, 8x, 16x` on tasks with known difficulty/depth tiers.

The expansion unit may be a generated residual block, reusable operator composition, temporary expert, or recurrent unroll. Only one expansion mechanism is allowed in V0.

## Measurements

- stored parameters;
- temporary generated parameters / logical modules;
- active parameters per step;
- total FLOPs and wall time;
- quality by task difficulty;
- quality-vs-compute frontier;
- whether expansion helps held-out deeper tasks rather than only training-like cases.

## Success gate

A positive result requires monotonic or clearly frontier-improving hard-task quality as temporary compute grows, while easy-task compute remains near the base model. A useful target is meaningful hard-task gain at `4x-16x` compute without changing stored parameters.

## Failure / stop rule

Reject the hypothesis if additional temporary capacity is ignored, collapses to duplicate functions, or quality saturates while compute rises sharply. Do not rescue V0 by adding persistent memory, new training data, a larger base model, or continual learning.

## Why it matters

This tests the core possibility that model size should not be a single fixed inference-time number. The relevant quantities become `stored capacity`, `active compute`, and `temporary reasoning capacity`.