# P-007 native task-context routing audit

Date: 2026-09-10  
Branch: `exp/track-runtime`

## Question

The task token is already present in the encoded input, but the hierarchical
router receives it only through the pooled recurrent query. An explicit small
task embedding may help the router distinguish task families without forcing a
fixed circuit assignment. Three opt-in variants were screened against the
same 300M ordered shared-route-key baseline, with balanced two-seed training
for 3,000 steps.

## Results

| Arm | Mean accuracy | Mean CE | Hard-task mean | Dead traffic |
|---|---:|---:|---:|---:|
| baseline, no task context | 68.451% | 0.99639 | 33.366% | 40.17% |
| task context in router and state | 68.047% | 1.02139 | — | 45.00% |
| router-only task context, scale 1 | 68.763% | 0.98239 | 33.203% | 48.95% |
| router-only task context, scale 0.25 | 68.294% | 0.98806 | 33.138% | 43.58% |

The scale-1 router-only arm gives a small overall accuracy and CE signal, but
its hard-task mean is lower and dead traffic is 8.78 points higher. Reducing
the context to 0.25 lowers the dead-traffic increase, but the accuracy becomes
`−0.156 pp` below baseline and hard-task mean remains lower. The full context
arm is worse because adding the task vector to every state update changes the
recurrent computation as well as routing.

## Decision

**No adoption.** Explicit task context is not a reliable quality fix for
P-007/P-003. It can make the router more task-specific, but that specialization
does not improve depth-2/3 computation and can fragment the bank. The
`task_context_scale` API is retained for controlled future experiments, with
the default behavior unchanged.

The next native experiment should target representation of partial results in
the recurrent state or an operation-aware circuit interface, not another
task-context scale sweep.

## Reproduction

The six runs are recorded in `results/runs/`:

- `ne300_task_context_s17_3000.json`
- `ne300_task_context_s18_3000.json`
- `ne300_task_context_router_only_s17_3000.json`
- `ne300_task_context_router_only_s18_3000.json`
- `ne300_task_context_router_only_scale025_s17_3000.json`
- `ne300_task_context_router_only_scale025_s18_3000.json`

