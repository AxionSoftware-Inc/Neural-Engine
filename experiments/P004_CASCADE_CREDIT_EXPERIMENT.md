# P-004 — Cascade-consistent on-policy credit experiment

Branch: `exp/expert-p004-cascade-shift`

Scope: **P-004 only**. The router architecture, circuit bank implementation,
model forward contract, default branch and historical result files are not
changed.

## Hypothesis

The current hard router is trained mainly through the route that was already
selected. A route change at recurrent step `t` changes the state consumed at
`t+1`, so evaluating a replacement while freezing the original suffix can give
the wrong credit target.

The treatment therefore uses the final corrected-output CE after this causal
trajectory:

```text
natural prefix -> changed route at step t -> changed recurrent state
              -> natural rerouting at t+1..T -> final corrected output
```

Only that cascade-consistent final CE supplies auxiliary route credit.
A fixed-suffix replay is retained as a diagnostic to measure cascade shift, but
never becomes the training target.

## What is unchanged

- `HierarchicalRouter` class and its candidate retrieval tree;
- candidate pool size and active circuit count;
- `MicroCircuitBank` body and circuit parameterization;
- recurrent state update and output correction;
- inference path and active parameter budget;
- existing configs/defaults and historical audits.

The benchmark uses `configs/ne_capacity_signal.yaml`: 32 circuits, candidate
pool 8, active 2, three recurrent steps, hard learned hierarchical routing.

## Treatment

Every 4 optimization steps, one recurrent step is selected cyclically.
For each example, one currently selected circuit is replaced by a different
circuit already present in the same candidate pool. This deliberately avoids
repeating P-001 candidate-retrieval work.

The replay plan uses `-1` for every non-intervened step. The existing model hook
therefore recomputes the prefix naturally, forces only the changed step, and
reroutes every suffix step from the changed recurrent state.

Let

```text
A = CE(natural final output) - CE(cascade-replayed final output)
```

If `A > 0`, the alternative pair should outrank the current pair. If `A < 0`,
the current pair is reinforced. Small `|A| < 0.01` probes are ignored. The
ranking loss is advantage-weighted and applied only to the existing router key
rows using the original step query detached from the recurrent graph.

This is intentionally not CRCA/P-002: auxiliary credit does not train circuit
rows. Circuits continue receiving only the normal task-loss gradient.

## Prefix / suffix diagnostics

Every 16 steps a second replay forces the complete original route except the
changed pair. Comparing it with the cascade replay reports:

- fixed-suffix final advantage;
- cascade-consistent final advantage;
- CE shift caused by rerouting the suffix;
- suffix route stability.

At held-out evaluation the benchmark also checks that the prefix is invariant
under a one-step intervention and measures suffix stability after the changed
state.

## Arms

For every seed all arms start from the exact same cloned initialization and
recreate the exact same main training data stream.

1. `control`: normal task-loss training only.
2. `cascade`: normal task-loss training plus cascade-consistent router credit.
3. `frozen_bank`: same cascade credit with circuit-bank parameters frozen;
   diagnostic only, not part of the adoption gate.

The auxiliary sampling RNG is separate from the model/data RNG, so treatment
probes do not change the main batch sequence.

## Metrics

Seed 17/18, 5000 steps each:

- held-out hard accuracy and CE;
- fresh train-distribution (`on_policy`) hard accuracy and CE;
- one-step within-candidate cascade oracle mean and p95 regret;
- prefix stability, suffix slot/exact route stability;
- mean cascade-vs-fixed-suffix CE shift;
- circuit usage/dead circuits;
- circuit-row and router-key gradient coverage;
- training seconds, samples/s, peak VRAM and treatment/control wall-clock ratio;
- planned training probe forward overhead.

Held-out regret enumerates every one-circuit replacement available inside the
current candidate pool at every recurrent step. The suffix always reroutes
naturally for this oracle.

## Pre-registered gate

The patch is accepted only if all guards pass across seed17/18:

1. held-out accuracy mean `>= +2 pp`, on-policy mean `>= +1 pp`, no seed
   negative on either accuracy delta;
2. mean held-out CE delta `<= -0.02`, no seed worse than `+0.02`;
3. mean and p95 cascade regret each improve by at least 10%;
4. circuit and router-key gradient coverage do not drop by more than `1/32`;
5. paired treatment/control wall-clock overhead `<= 1.50x`;
6. counterfactual prefix route stability `>= 99.9%`.

A CE-only improvement is not sufficient. If any gate fails, the result is
`REJECTED` and P-004 remains `ACTIVE`. With `--update-problems`, the benchmark
appends the rejection record to `problems.md`; it does not rewrite historical
result files.

## Reproduction

```bash
python -m pytest tests/test_p004_cascade_credit.py -q

python benchmark_p004_cascade.py \
  --config configs/ne_capacity_signal.yaml \
  --steps 5000 \
  --device cuda \
  --seeds 17 18 \
  --update-problems
```

Structural CPU smoke test:

```bash
python benchmark_p004_cascade.py --device cpu --smoke --seeds 17
```

New outputs are written only to:

- `results/P004_CASCADE_CREDIT_SEED17_18.json`
- `results/P004_CASCADE_CREDIT_SEED17_18.md`
- `results/checkpoints/p004_cascade/`
