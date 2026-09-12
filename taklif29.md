# taklif29.md — Persistent Structural Growth: allocate new learned parameters only when new knowledge requires them

Status: **independent lifelong-capacity proposal; temporary reasoning expansion is out of scope for V0**

> **SCOPE:** Test whether a trained model can preserve a stable core and permanently add small new modules when incoming knowledge cannot be represented well by existing modules. This proposal is about stored model growth, not temporary inference-time expansion.

## Hypothesis

Let the initial model be `theta_core`. For new knowledge pack `D`, estimate whether existing capacity is sufficient. If not, allocate a new module `phi_new` and train only that module plus a small routing interface:

`theta_t = [theta_core, phi_1, ..., phi_t]`.

The model therefore may grow from, for example, `27B -> 27.1B -> 28B -> 32B` as genuinely new information accumulates, without rewriting the entire core each time.

## V0 experiment

Use a small frozen base model and sequentially introduce disjoint knowledge domains. For each domain compare:

1. ordinary fine-tuning of the whole trainable adapter/model;
2. fixed-capacity continual adapter;
3. growth-enabled expert allocation with the old modules frozen.

Growth decisions must use a fixed threshold derived from held-out utility, not manual intervention.

## Measurements

- stored parameter growth per domain;
- active parameters per query;
- new-domain quality;
- old-domain retention;
- routing accuracy to new modules;
- parameter efficiency: quality gain per added parameter;
- cumulative training compute;
- catastrophic forgetting.

## Success gate

A positive result requires sequential knowledge acquisition with lower old-task regression than conventional fine-tuning and better cumulative quality than a same-size fixed-capacity control. Added parameters must show causal utility under ablation.

## Failure / stop rule

Reject if modules grow without measurable new capability, if routing cannot reliably use them, or if storage grows nearly linearly with raw corpus size while knowledge quality remains poor.

## Why it matters

This is the mechanism that would make a pretrained model genuinely become larger over its lifetime because it has acquired new information, rather than merely generating temporary reasoning compute.