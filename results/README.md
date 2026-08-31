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
