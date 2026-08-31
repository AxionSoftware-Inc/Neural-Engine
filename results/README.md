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
