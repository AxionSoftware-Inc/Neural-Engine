# Compact coupled probe router: audit and rejection

Date: 2026-09-06. Decision: **reject for adoption**. No default router,
NeuralEngineV0 API, model body, circuit bank, or existing checkpoint was changed.
The implementation remains an opt-in, reproducible negative experiment.

## What the V0.178–V0.180 audit established

- V0.178: selection headroom 0.262/0.276 CE and additional retrieval headroom
  0.195/0.171 CE. These are one-decision oracles, not global trajectory oracles.
- V0.179: full-bank key scoring alone did not solve learning; FlatRouter reached
  44.89% versus 47.18% hierarchical, with 34.4%/37.5% dead circuits in the full run.
- V0.180: ProbeRouteRouter avoided most dead circuits, but mean CE improvement
  did not translate into robust hard accuracy. Its 240-example evaluation and
  repeated eight training batches did not establish generalization/scaling.

Implementation findings in `neural_engine/router.py` and `probe_route_frozen.py`:

1. Separate D-dimensional retrieval and utility projections/key tables can
   disagree. The selector can learn an outside circuit's utility without that
   update reaching the retriever.
2. Retrieval uses positive-only outside/double-probe inclusion labels. Its
   masked loss is averaged over the entire batch, then multiplied by 0.1;
   effective signal strength therefore depends on the positive-probe rate.
   A pair improving the current route is not necessarily better than the
   candidate oracle. The inclusion surrogate is not exact retrieval regret.
3. Old pair embeddings start with standard deviation 0.02. Their products are
   initially small; this can weaken the early interaction signal. This is a
   mechanism hypothesis, not a demonstrated root cause.
4. V0.180 recomputes the ordinary forward and replays the entire batch including
   unprobed rows. The new benchmark reuses ordinary CE and replays only probed
   examples. Both compared routers receive the same implementation improvement.
5. `NeuralEngineV0.parameter_report()` does not include the complete probe-router
   projections/tables in its active estimate. The benchmark reports explicit
   conservative touched-parameter bounds instead; the model body was not patched.
6. The old probe coverage diagnostic ignores `coverage_temperature` and uses
   full-bank normalization even under restricted capacity. The opt-in subclass
   respects temperature and the currently reachable capacity.
7. Old exploration repairs repeated random IDs by incrementing one ID, creating
   a sampling bias. The new subclass samples distinct pairs without replacement.

These findings do not prove that a compact coupled router will improve quality.

## Minimal implementation

`neural_engine/coupled_probe.py` subclasses the existing routing interface:

- one 128-to-32 query projection, non-affine input LayerNorm and GELU;
- one shared 32-dimensional circuit-key table for retrieval and utility;
- rank-8 symmetric pair interaction, with a zero-initialized interaction query;
- top-8 key retrieval, 28 inexpensive pair scores, exactly two executed bodies;
- uniform pair weights and unit gain, matching the old ProbeRouteRouter;
- no task-ID targets, attention, Transformer, new body layers, or dense bank pass;
- coverage, candidate statistics, capacity restriction and exploration support.

The router has **5,632 parameters**, versus **42,240** for ProbeRouteRouter (7.5x
fewer). Smaller parameter storage is not a claim of 7.5x lower latency.

Signed probe targets are final CE(current) minus final CE(alternative). The loss
combines Huber difference regression, cost-weighted hard ranking, and a 0.25
coarse additive difference-regression term. The latter reaches the shared
retrieval keys for positive AND negative probes. The additive target is a
deliberately limited surrogate; interaction can invalidate marginal rankings.
It is tested through exact retrieval metrics, not assumed correct.

To install explicitly for an experiment, instantiate `CoupledProbeRouter` and
assign it to an existing compatible model's `router`, moving it to the model's
device. No factory default or existing router variant was replaced. Normal
NeuralEngineV0 `forward`, forced-route replay, and statistics APIs are unchanged.

## Paired protocol actually run

- Physical bank: E=32, K=2, M=8, three recurrent steps, parallel circuits.
- Frozen hierarchical checkpoints: `capacity_audit_c32_global_s17.pt` and s18.
- Every arm retains identical non-router tensors, verified by SHA-256 before
  and after training. No circuit-body or controller weights are trained.
- 2,000 updates: 200 teacher-route imitation initialization, 1,800 probe updates.
  Same initialization rule per seed, LR=3e-4, AdamW weight decay=1e-4, clip=1.
- Fresh balanced training batch each update, 8 examples/task = 120 examples.
  No reuse of a fixed eight-batch corpus. Seeded sampling has identical probe
  kind/count budgets across arms, although on-policy circuit identities differ.
- Per example: 75% no probe, 10% inside, 10% outside, 5% double swap. One
  recurrent decision per batch is sampled uniformly. Alternatives are sampled
  uniformly in each eligible set. Probe branches are no-grad; this frozen test
  does not claim to solve expert starvation during joint training.
- Replay preserves the actual prefix IDs, weights and gains, overrides one
  pair, and re-routes the suffix on the changed state. Prefix computation is
  recomputed through the public API and charged, not claimed to be cached.
- Held-out CE/accuracy/usage: 3,840 examples per seed, separate RNG from training.
- Evaluation-only exhaustive oracle: 60 different held-out examples per seed,
  all 496 pairs at each of three decisions. All arms are queried on the same
  hierarchical reference states and use the same hierarchical suffix policy.
  These diagnostic regrets are NOT each arm's on-policy trajectory oracle.
- Candidate recall means the pool contains both members of at least one
  CE-optimal pair (ties within 1e-6). p95 regret is chosen-pair CE minus full-bank
  best CE, over 180 reference decisions/seed. No oracle costs enter training.
- CPU, two threads, PyTorch 2.6.0+cu124; no GPU benchmark/training was launched.
  Latency: ten warmups and fifty repeats, public model forward including stats.
  Arm training order is reversed on seed18. Timing is a small local screen,
  not an isolated systems benchmark.

Arms:

- `old`: ProbeRouteRouter with its original positive-only inclusion objective.
- `old-signed`: SAME old architecture with the new signed objective, to separate
  objective changes from architecture changes.
- `coupled`: new compact router and signed objective.
- `hierarchical`: original frozen checkpoint reference, no refit.

V0.180's published numbers are not directly mixed into these results: its data
budget/evaluation set differ. All comparisons below use the new paired protocol.

## Measured quality

| Seed | Router | CE | Hard accuracy | Candidate recall | Selection regret CE | Retrieval regret CE | p95 regret CE | Dead/32 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | hierarchical | 2.43511 | 48.698% | 8.333% | 0.36486 | 0.22161 | 2.76048 | 0 |
| 17 | old | 2.48253 | 47.969% | 17.222% | 0.39801 | 0.18687 | 2.79507 | 0 |
| 17 | old-signed | 2.46600 | 45.964% | 5.556% | 0.21448 | 0.24036 | 1.71160 | 0 |
| 17 | coupled | 2.42908 | 47.839% | 8.889% | 0.34367 | 0.19253 | 1.97437 | 0 |
| 18 | hierarchical | 2.61220 | 45.417% | 18.333% | 0.35312 | 0.15205 | 2.26333 | 0 |
| 18 | old | 2.57663 | 45.365% | 18.333% | 0.30315 | 0.15899 | 1.87519 | 0 |
| 18 | old-signed | 2.62838 | 42.552% | 2.222% | 0.25831 | 0.19283 | 1.66985 | 0 |
| 18 | coupled | 2.52295 | 45.833% | 10.000% | 0.29145 | 0.15247 | 1.71300 | 0 |

Coupled minus old:

- Mean CE: **-0.05357**; improves both seeds.
- Mean hard accuracy: **+0.169 percentage points**, not +2 points.
- Seed17 accuracy **-0.130 pp**; seed18 **+0.469 pp**.
- Candidate recall **-8.333 pp in each seed**.
- p95 regret improves 29.36%/8.65%; seed18 misses the 10% gate.

The old-signed control also loses candidate recall and hard accuracy. This
supports caution about signed additive retrieval targets, not a claim that
probe feedback is universally wrong. Reducing average/tail CE cost can coexist
with missing the very best pair and failing to improve argmax decisions.

## Parameters and CPU latency

Active parameter counts below are conservative per-decision bounds, including
shared body parameters, router parameters touched, and two circuit rows. They
are not FLOPs, physical memory traffic, or sums multiplied by recurrent steps.
Each arm activates **4,352 circuit-body parameters per decision**.

| Router | Router parameters | Active parameters bound/decision | Seed17 batch median/p95 ms | Seed18 batch median/p95 ms |
|---|---:|---:|---:|---:|
| hierarchical | 8,224 | 233,440 | 6.677 / 8.451 | 6.750 / 8.671 |
| old | 42,240 | 270,336 | 7.074 / 9.161 | 6.960 / 8.882 |
| old-signed | 42,240 | 270,336 | 6.925 / 7.939 | 7.079 / 8.448 |
| coupled | 5,632 | 233,728 | 6.845 / 8.167 | 6.951 / 8.466 |

Batch size is 120. Single-example median latency old/new: 2.114/1.975 ms on
seed17 and 2.104/2.058 ms on seed18. Overall batch latency is essentially similar.
Old/new refit time: 35.77/32.73 seconds and 35.30/33.17 seconds.

All three trained arms forwarded exactly 293,714 example-trajectories on seed17
and 293,843 on seed18, including imitation and probe replay. Every trajectory
uses three steps with two body rows. Probe targets do not densely evaluate all
32 bodies. The exact all-pair evaluation audit is separately accounted work.

## Predeclared gate and decision

Acceptance required mean accuracy +2 pp over old; both seeds positive accuracy
and CE improvement >=0.05; p95 regret reduction >=10%; candidate recall not
worse; median batch latency <=1.25x; dead circuits <=3; and accuracy at least the
hierarchical reference on both seeds.

**FAIL: reject for adoption.** Accuracy, candidate recall, seed18 p95 regret,
and seed17 hierarchical non-inferiority gates fail. Do not change the default,
launch joint training, or scale to 300M on the strength of this result. The
compact implementation is smaller, not a demonstrated quality solution.

The oracle audit is small and no third seed or confidence-interval study was
run. That limits inference, but cannot justify promoting a failed screen.

## Reproduce and tests

```powershell
python benchmark_probe_router.py --steps 2000 --imitation-steps 200 --device cpu --threads 2 --eval-batches 32 --audit-examples-per-task 4 --output results/runs/coupled_probe_reproduction.json
python -m pytest tests/test_coupled_probe.py tests/test_router.py tests/test_forward.py tests/test_training_controls.py tests/test_optimizer.py -q
```

Use a new output name; the script refuses to overwrite an existing result.
The executed command used `results/runs/coupled_probe_paired_cpu_s17_s18.json`.
Raw JSON includes per-seed metrics, per-circuit usage, parameter bounds, latency
quantiles, training compute counts, frozen-body digests and the reject decision.
Run JSONs follow the repository's existing gitignore convention.

**38 tests passed.** Tests cover shape/selection/statistics contracts, symmetry,
capacity validation, differentiable coverage, no duplicate exploration IDs,
outside-key gradient and detached targets, empty probes, forced replay/sentinel
parity, exactly two circuit IDs per body call, frozen-body training across all
three arms, oracle cost correctness/decomposition, and rejection of CE-only gains.
