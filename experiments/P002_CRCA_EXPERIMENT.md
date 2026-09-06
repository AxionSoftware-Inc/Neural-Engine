# P-002 expert experiment — Causal Responsibility Credit Assignment (CRCA)

Branch: `exp/expert-p002-circuit-specialization`  
Base lineage: `066c2a910ddd2cd96c65711eb1c098638374cb5d`  
Scope: **P-002 only** — circuit-bank specialization / sparse credit assignment.

## Why CTCA v1 was tightened before the long screen

The first expert patch used Counterfactual Tournament Credit Assignment (CTCA):
periodically probe a few starved rows, choose one batch-level winner, and give
only that row final-task-loss gradient. It preserved the router contract, but
one design weakness remained before spending two 5000-step seeds on it:
batch-average winner training can turn a circuit into another generic backup
rather than a reusable niche specialist.

The repository history makes this distinction important:

- V0.175: fixed controlled allocation beats learned 32-bank routing by `+4.68 pp`
  held-out on the two-seed mean, but raw capacity itself scales weakly.
- V0.176: family partitions and training-only soft routing were rejected.
- V0.177: arbitrary task-to-route target supervision reduced entropy but did
  not improve final quality.
- V0.95: high circuit/factor coverage did not correlate monotonically with
  accuracy, so "make every row active" is not a sufficient objective.
- `COUPLED_PROBE_ROUTER_AUDIT.md`: a much smaller router improved CE and
  efficiency but hard accuracy changed only `+0.169 pp`, while candidate recall
  fell `-8.33 pp` on both seeds. Another router rewrite is therefore out of scope.
- shared residual training already supplied dense common gradient without
  teaching the sparse rows useful specialization.
- factorized banks have positive reuse results in a different typed-register
  protocol, so simply replacing this 32-row bank with another factorized bank
  would not isolate P-002.

CRCA therefore changes **credit granularity**, not routing architecture or
circuit parameterization.

## Hypothesis

A starved circuit should receive gradient only on examples where substituting it
into the exact on-policy trajectory *causally reduces final task loss*.

If each example assigns responsibility to its own best shadow circuit, different
rows can acquire different niches even inside the same minibatch. This avoids
both arbitrary task IDs and uniform coverage.

## Training event

Every `credit_interval` steps:

1. keep the live route, candidate retrieval, route weights and route gains fixed;
2. rotate through one `(internal_step, active_slot)` role;
3. choose a small shadow set by **lowest on-policy usage**;
4. for every shadow circuit, evaluate only examples where that circuit is absent
   from the original full trajectory;
5. compute per-example final-CE advantage:
   `baseline_final_CE - substituted_final_CE`;
6. each example selects the shadow circuit with the highest advantage;
7. examples below `min_advantage` receive **no auxiliary credit**;
8. each circuit is replayed only on the examples for which it owns responsibility;
9. auxiliary CE is importance-weighted by clipped causal advantage;
10. `torch.autograd.grad` is requested for the circuit bank, then **only the
    responsible circuit row** (`down/up/bias`) is added to the main gradient;
11. router, controller, encoder and output auxiliary gradients are discarded.

A credit event may update multiple rows if different examples have different
causal winners.

## Why this is not coverage, exploration or another router

Candidate priority is `(on_policy_usage, shadow_update_count, random_tie)`.
Shadow update count is only a tie breaker. A circuit with the lowest real
on-policy use remains the first probe target even after prior shadow updates.
This prevents CRCA from degenerating into round-robin coverage.

The normal training/inference route is never replaced. There is no new router
class, no candidate-pool change, no route exploration, no coverage regularizer,
no task-ID target and no soft mixture.

## Held-out functional specialization audit

Routing NMI alone cannot validate this experiment because the router is
intentionally unchanged. The benchmark therefore adds a second, gradient-free
held-out audit.

For each audited balanced batch:

- one of all six `(3 steps × 2 slots)` roles is selected, rotating over roles;
- every one of the 32 circuit rows is tested as a forced substitution where it
  was absent from the original trajectory;
- the best positive final-CE substitution receives held-out responsibility;
- task↔responsible-circuit NMI, usage-weighted specialization, purity,
  positive-responsibility fraction, responsible circuit count and mean/median
  positive CE advantage are recorded.

This metric asks whether the **circuit bodies themselves** acquired distinct
useful functions, independent of whether the current router can retrieve them.
It is evaluated on held-out batches and does not reuse training responsibility
counts for the acceptance gate.

## 20M / 32-bank protocol

The existing opt-in `configs/ne_p002_20m_32.yaml` remains the controlled model:

- approximately 20M stored parameters;
- exactly 32 independent `MicroCircuitBank` rows;
- candidate pool 8;
- active circuits 2;
- 3 internal recurrent steps;
- existing `HierarchicalRouter`;
- no route exploration;
- no coverage regularizer;
- no adaptive halting.

Control and CRCA start from the exact same initialization for each seed and
receive the same balanced training stream. Default model/configs are not
changed.

## Gradient order

For a treatment step:

1. normal on-policy forward;
2. CRCA no-grad counterfactual probes when scheduled;
3. row-local counterfactual gradients are materialized but not yet applied;
4. normal task loss backward computes the ordinary model/router/controller
   gradient;
5. diagnostics sample one circuit's normal gradient;
6. only responsible shadow row gradients are added;
7. one global gradient clip;
8. normal optimizer step.

This makes the causal auxiliary term additive to the existing training rule
without giving auxiliary gradients to the router.

## Training vs inference cost

Default settings:

- `credit_interval=8`
- `credit_candidates=4`
- `credit_weight=0.25`
- `credit_min_eligible=16`
- `credit_min_responsible=4`
- `credit_min_advantage=0.02`
- `credit_advantage_clip=1.0`

The planned probe upper bound is `4/8 = 0.5` candidate probe events per normal
step. Up to four row-local differentiable replays can occur at an event, but
they operate only on each row's responsibility subset. The JSON records actual
probe examples and actual differentiable replay examples normalized by normal
training examples.

Inference path and inference active parameters are exactly unchanged.

## Pre-registered gate

Accept P-002 CRCA if either:

1. **quality gate:** mean seed17/18 hard accuracy is at least `+2.0 pp`, with no
   seed worse than `-1.0 pp`; or
2. **held-out functional specialization gate:** versus the paired control,
   mean counterfactual task↔circuit NMI improves by at least `+0.05`, mean
   counterfactual specialization by at least `+0.05`, mean positive final-CE
   advantage by at least `+0.01`, neither seed's advantage delta is below
   `-0.005`, and dead-route fraction worsens by no more than `1/32`.

The second gate deliberately requires three held-out causal signals together;
training responsibility logs alone cannot pass the experiment.

If both gates fail, the runner writes `REJECTED` and, when
`--update-problems-on-reject` is supplied, appends the negative result to
`problems.md` without modifying historical entries.

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

Structural smoke only:

```bash
python benchmark_p002_specialization.py \
  --config configs/ne_p002_20m_32.yaml \
  --device cpu \
  --smoke \
  --seeds 17
```

Do not adopt from a smoke result. The authoritative decision is the paired
seed17/18 long screen and independent local verification by the main agent.
