# taklif31.md — Two-Timescale Elastic Lifelong Model: persistent knowledge growth plus temporary reasoning expansion

Status: **integration proposal; do not test before the component gates in taklif24 and taklif29 pass independently**

> **SCOPE:** Combine two already-validated mechanisms only after they work separately: (1) persistent structural growth for genuinely new knowledge and (2) temporary task-conditioned expansion for difficult reasoning. This proposal must not be used to hide failure of either component.

## Hypothesis

Model state has three distinct capacities:

- `P_stored`: persistent learned knowledge/capability;
- `P_active`: parameters/modules used in the current step;
- `P_effective`: temporary reasoning graph/cumulative logical compute used to solve the current task.

A lifelong model may grow slowly in stored capacity as new information arrives while expanding much more aggressively during hard reasoning:

`27B stored -> 32B stored over time`,

but for one difficult task:

`32B stored -> 100B/300B-equivalent temporary effective compute -> 32B stored`.

## V0 experiment

Use a small modular model with several sequential knowledge additions. After each addition, evaluate easy and hard reasoning tasks under adaptive temporary compute budgets.

Controls:

1. fixed stored model + fixed compute;
2. persistent growth only;
3. temporary expansion only;
4. persistent growth + temporary expansion.

All four must share the same data and total evaluation protocol.

## Measurements

- stored parameter growth over knowledge episodes;
- active and temporary compute per difficulty tier;
- new-knowledge quality;
- old-knowledge retention;
- hard reasoning quality;
- quality per stored parameter;
- quality per inference FLOP;
- whether newly learned modules are actually reused during temporary reasoning.

## Success gate

The combined system must beat both single-mechanism controls on a Pareto frontier, not merely spend more compute. New stored knowledge should improve future reasoning, while temporary expansion should improve hard-task quality without permanently bloating storage.

## Failure / stop rule

Reject integration if one mechanism dominates and the other adds cost without capability, or if the controller cannot distinguish when to learn permanently versus when to spend temporary reasoning compute.

## Why it matters

This is the full form of a self-growing yet elastic model, but it is intentionally **not** the first experiment. Each underlying claim must first survive independent falsification.