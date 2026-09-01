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
