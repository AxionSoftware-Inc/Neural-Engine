# P-002 expert experiment — Counterfactual Tournament Credit Assignment (CTCA)

Branch: `exp/expert-p002-circuit-specialization`  
Base commit: `066c2a910ddd2cd96c65711eb1c098638374cb5d`  
Scope: **P-002 only** — circuit-bank specialization / sparse credit assignment.

## Status before local benchmark

Implementation and benchmark protocol are prepared. The expert connector used to
write this branch has no CUDA/local-repository execution environment, so no
seed17/18 number is invented here. `benchmark_p002_specialization.py` is the
authoritative paired runner; it writes JSON/Markdown results and can append a
`REJECTED` entry to `problems.md` when the pre-registered gate fails.

The existing default model, router implementation, historical results and audit
files are unchanged.

## Why this experiment

P-002 has a self-reinforcing starvation loop: a rarely selected circuit receives
little task gradient, remains weak, and therefore keeps losing future route
selection. Additional stored capacity then fails to become useful capacity.

Several simpler paths are deliberately not repeated:

- full-bank router scoring already failed to fix quality;
- coverage-aware training has already been explored;
- random route exploration produced only a weak signal;
- controlled task allocation demonstrated headroom but hard-codes the answer.

CTCA asks one narrower causal question: **if an under-trained independent circuit
receives task-aligned gradient without changing the router, can useful
specialization emerge?**

## What changes, in order

1. **Circuit body: unchanged.** `MicroCircuitBank` stays identical so a positive
   result cannot be attributed to a body rewrite.
2. **Controller/output: unchanged for auxiliary learning.** Persistent state,
   encoder and output head still receive the normal on-policy loss only.
3. **Router: unchanged.** No router class, score, retrieval rule, route loss or
   inference path changes.
4. **Sparse credit assignment: training-only.** Every configured interval a
   small tournament probes starved circuits with forced-route replay and applies
   one row-local auxiliary update.

At a credit event the algorithm:

1. cycles to one `(internal_step, active_slot)` role;
2. prioritizes candidates with few shadow updates and low on-policy usage;
3. keeps only examples where a candidate is absent from the entire original
   trajectory;
4. replays the exact on-policy route, weights and route gains while replacing
   only that role;
5. ranks candidates by final task-loss advantage;
6. replays the winner with gradients enabled;
7. calls `torch.autograd.grad` only for `circuits.down/up/bias`;
8. discards every non-winning circuit row and every router/controller/output
   auxiliary gradient;
9. adds only the winning circuit-row gradient to the normal on-policy gradient.

The best candidate is allowed bootstrap credit even if all candidates are still
worse than the current route early in training. Requiring positive advantage
from a starved, untrained row would recreate the starvation loop. The tournament
still avoids uniformly training every dormant circuit on every sample.

## Not route exploration / not P-004

The live route used by the main loss never changes. Counterfactual routes are
training probes only and do not become inference actions. The original suffix
route is frozen too; CTCA does not learn a new cascade policy or route target.
The final loss is used solely to assign a row-local circuit training signal.

## 20M / 32-circuit controlled model

`configs/ne_p002_20m_32.yaml` keeps the P-002 capacity-signal route contract:

- 32 circuits;
- candidate pool 8;
- active circuits 2;
- 3 internal steps;
- learned hierarchical router;
- no route exploration;
- no coverage regularizer;
- no adaptive halting.

State/circuit dimensions are enlarged so stored parameters are near 20M while
the bank contains exactly 32 independent circuits. The benchmark fails fast
unless `18M <= total_params <= 22M` and `num_circuits == 32`.

This is an **opt-in experiment config**. Existing `ne_20*` configs, defaults and
historical benchmarks are not replaced.

## Paired seed17/18 protocol

For each seed:

- instantiate one initial model;
- clone the exact same initial `state_dict` into control and CTCA arms;
- recreate the same balanced training generator;
- train equal optimizer steps;
- keep the learned router identical;
- evaluate on the same balanced protocol;
- report accuracy, loss, task↔circuit NMI, usage-weighted specialization, task
  purity and dead circuits.

Per-circuit diagnostics include on-policy usage, examples touching each circuit,
cyclic samples of main-loss gradient norm/non-zero rate, shadow probe examples,
shadow update events/examples, mean counterfactual advantage and auxiliary
row-gradient norm.

Gradient norms sample one circuit per step instead of scanning all ~20M
parameters every step. At 5000 steps/32 circuits that is about 156 observations
per circuit without making diagnostics the dominant compute cost.

## Training vs inference cost

Default CTCA settings are interval `8`, candidates `4`, weight `0.25`.
The conservative planned overhead report is `(4+1)/8 = 0.625` extra forward
equivalents and `1/8 = 0.125` extra circuit-only backward equivalents per
normal training step. This is training-only cost. Inference route and active
parameters are exactly the control model's.

## Pre-registered gate

Accept if either:

1. mean seed17/18 accuracy improves by at least `+2.0 pp`, with neither seed
   worse than `-1.0 pp`; **or**
2. mean task↔circuit NMI delta is at least `+0.05`, mean usage-weighted
   specialization delta is at least `+0.05`, and dead-circuit fraction worsens
   by no more than `1/32`.

If neither gate passes, decision is `REJECTED`. Running with
`--update-problems-on-reject` appends a new rejection record to `problems.md`
without deleting or rewriting historical audit/rejection text.

## Reproduction

```bash
python -m pytest tests/test_p002_credit.py -q

python benchmark_p002_specialization.py \
  --config configs/ne_p002_20m_32.yaml \
  --steps 5000 \
  --device cuda \
  --seeds 17 18 \
  --update-problems-on-reject
```

Structural smoke run only:

```bash
python benchmark_p002_specialization.py \
  --config configs/ne_p002_20m_32.yaml \
  --device cpu \
  --smoke \
  --seeds 17
```

The full runner creates new result/checkpoint files under `results/`; it never
rewrites historical result or audit files. Do not change the default model based
on smoke output. Adoption is allowed only after the paired 5000-step seed17/18
gate is independently verified.
