# Results

Each meaningful run is saved as a JSON file under `results/runs/`. Milestone
reports should record the exact command, commit SHA, hardware, quality, active
parameter estimate, routing statistics, and whether the result is a failure or
positive signal.

Useful trained checkpoints and their reproducible GPU measurements are
documented in `CHECKPOINTED_INFERENCE.md`. Checkpoint binaries live only in the
local `results/checkpoints/` directory and are ignored by Git.

Router utilization and task-overlap analysis is documented in
`ROUTE_STABILITY.md`; the reproducible entry point is `analyze_routes.py`.

Controlled one-operand and one-operation route sensitivity is documented in
`COUNTERFACTUAL_ROUTE_SENSITIVITY.md`; use `analyze_counterfactual_routes.py`
to reproduce it.

The stronger causal route-replay test is documented in
`ROUTE_REPLAY_CAUSALITY.md`; use `analyze_route_replay.py` to reproduce it.

Route replacement at 0/25/50/100% global and within-task rates is documented
in `ROUTE_SWAP_ABLATION.md`; use `analyze_route_ablation.py` to reproduce it.

The active-circuit k=4/8/16 quality and latency sweep is documented in
`ACTIVE_CIRCUIT_BUDGET.md`; use `analyze_active_budget.py` to reproduce it.

Scratch training and second-seed validation for k=4 versus k=8 is documented
in `V0_12_MULTI_SEED_BUDGET.md`.

The optional coverage-aware low-k router regularizer is documented in
`V0_13_COVERAGE_REGULARIZER.md`; it improves k=4 bank utilization but does not
yet justify changing the k=8 default.

The V0.12 stage-supervision re-test is documented in
`V0_14_STAGE_SUPERVISION.md`; it improves depth-3 accuracy but lowers overall
and held-out quality, so it remains an optional composition-focused recipe.

The current numeric NE-20/NE-50/NE-100 fixed-active scaling test is documented
in `V0_15_NUMERIC_CAPACITY_SCALING.md`; it keeps the active path near 2.04M
parameters while growing stored capacity to 100M.

The recurrent input-reinjection ablation is documented in
`V0_16_INPUT_REINJECTION.md`; reducing reinjection slightly helps depth-3 but
does not improve overall or held-out quality.

The optional explicit gated memory/write ablation is documented in
`V0_17_GATED_MEMORY_WRITE.md`; it helps one held-out measurement but increases
active cost and lowers full-benchmark quality.

The new arithmetic composition benchmark and its all-pairs learnability
control are documented in `V0_18_COMPOSITION_BENCHMARK.md`. Both models remain
underfit at 5,000 steps, so the result is a diagnostic and not a final
generalization claim.

The fair numeric Transformer re-test is documented in
`V0_19_FAIR_NUMERIC_BASELINE.md`. It shows that the old plain-embedding
Transformer control understated dense quality; NE's strongest remaining claim
is active-compute/throughput efficiency, not unconditional accuracy.

The first route-indexed batched `LazyAdamW` prototype is documented in
`V0_20_LAZY_ADAMW.md`. It stays within 0.32 held-out points of dense AdamW and
is only about 4.7% slower at 20M, but does not yet implement RAM offload or
cache/prefetch. The same optimizer remains within 0.47 points of the 100M
dense reference in a single seed and passes a 300M early feasibility screen.

The CPU-RAM LRU circuit paging prototype is documented in
`V0_21_CPU_CACHE_PAGING.md`. A full working-set cache reaches 97.61% hit-rate
and reduces 100M benchmark H2D traffic from 6.99 GB to 167 MB, but still needs
pinned-memory, asynchronous-prefetch, and batched-kernel work.

The 300M/500M sparse scaling and route-coverage study is documented in
`V0_22_SCALE_300M_500M.md`. The 300M model reaches 56.80% after 5,000 steps;
500M fits on a 12 GiB RTX 3060 but reaches only 44.32% without coverage and
44.11% with coverage, so the larger bank is feasible but not yet a useful
quality scaling step.

The parent-based capacity-growth experiment is documented in
`V0_23_CAPACITY_GROWTH.md`. Cloning new 500M circuit rows from a trained 300M
parent and warming up on the parent routing geometry raises mean held-out
accuracy to 52.34% versus 45.63% for scratch 500M, across two seeds. This is
a positive warm-start signal, not yet a scratch-training comparison.

The third-seed validation is documented in `V0_24_GROWTH_THIRD_SEED.md`.
Across three seeds, parent-based growth reaches 51.91% ± 0.83 held-out
accuracy versus 45.63% for scratch 500M, confirming the direction of the
improvement while preserving the warm-start caveat.

The V0.25 composition scaling falsification is documented in
`V0_25_COMPOSITION_GROWTH.md`. On the hidden operation-order benchmark,
300M and 500M remain underfit; 500M scratch reaches 10.16% held-out accuracy,
while parent-growth reaches 9.38%. This is a NO-GO for a 700M/1B quality jump
until circuit credit assignment or composition is improved.

The V0.26 training-only route exploration test is documented in
`V0_26_ROUTE_EXPLORATION.md`. A 5% random tree-branch exploration probability
raises 300M held-out accuracy from 7.03% to 11.72% and 500M growth from 9.38%
to 10.94% in one seed under the original small random-batch evaluation. The
later deterministic-grid audit is the primary quality reference; 700M/1B
remains blocked pending stronger composition results.

The V0.27 audited curriculum scaling study is documented in
`V0_27_CURRICULUM_SCALE_AUDIT.md`. With deterministic 4,096-example-per-pair
grids, 300M curriculum averages 13.79% ± 0.96 across two seeds, while 500M
curriculum growth averages 12.77% ± 0.24. Curriculum is a weak positive, but
500M still does not beat 300M, so 700M/1B quality scaling remains blocked.

The V0.28 typed-register architecture and 20M/50M/100M/300M scale audit is
documented in `V0_28_TYPED_REGISTER_SCALE.md`. Explicit operand/partial/final
registers and typed serial circuit execution raise hidden-pair grid accuracy
to 65.56% at 20M and 89.87% at 100M; 300M reaches 89.20% and is not better
than 100M. The result is a strong architecture positive, but not a reason to
scale to 700M/1B before route sharing and a second-seed validation.

The V0.29 routing specialization and seed audit is documented in
`V0_29_ROUTING_SPECIALIZATION_AUDIT.md`. Partitioned routing reaches 87.01%
on one 20M seed but 61.46% on the second and 68.65% at 100M. A compressed
32-dimensional route context reaches 71.75% at 20M but 61.22% at 100M. These
results reject partitioning as a reliable default and keep the larger-scale
capacity claim open.

The V0.30 active-budget audit is documented in
`V0_30_ACTIVE_BUDGET_AUDIT.md`. Doubling active circuits from 8 to 16 at 20M
does not improve all-pairs accuracy (57.67% → 57.06%) or hidden-pair accuracy
(65.56% → 63.35%), and costs about 1.5× more local training time. The next
target is credit assignment and dataflow supervision, not wider active paths.
