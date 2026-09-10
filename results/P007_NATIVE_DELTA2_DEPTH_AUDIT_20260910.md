# P-007 native correction amplitude and depth specialization audit

Date: 2026-09-10  
Branch: `exp/track-runtime`

## Question

The 10k capacity diagnostic showed that selected circuit corrections are a
small part of the recurrent state, especially on depth-2/3 examples. Two
checks were run without forcing routes:

1. a two-seed 300M training screen with `circuit_delta_scale=2.0`;
2. a no-training depth specialization diagnostic on matched 10k 300M/500M
   checkpoints across uniform and edge-value probes.

## Delta-scale screen

The `delta=2` arm changed only the multiplier applied to the selected circuit
correction in the state update. Router, factor bank, active budget, optimizer,
and balanced training protocol stayed the same. Both seeds ran for 3,000 steps.

| Arm | Mean accuracy | Mean CE | Hard-task mean |
|---|---:|---:|---:|
| shared route-key baseline, scale 1 | 68.451% | 0.99639 | 33.366% |
| shared route-key, scale 2 | 68.620% | 1.00105 | 33.398% |
| delta | **+0.169 pp** | **+0.00466** | **+0.033 pp** |

The accuracy difference is too small, hard-task quality is unchanged, and CE
gets worse. It does not meet the P-007 quality gate. The screen is therefore
**REJECTED as a correction/state-path fix**. The implementation remains only
as an opt-in control.

## Depth specialization diagnostic

For each selected step, the diagnostic measured the mean pair cosine of the
selected circuit outputs and the norm of the weighted selected route relative
to the recurrent query. The 10k checkpoints were evaluated with 32 examples
per task. These are structural diagnostics, not causal proof.

### Uniform `[0,63]` probe, two-seed means

| Depth | 300M selected pair cosine | 500M selected pair cosine | 300M route/query norm | 500M route/query norm |
|---:|---:|---:|---:|---:|
| 1 | 0.255 | 0.241 | 9.56% | 7.45% |
| 2 | 0.278 | 0.272 | 6.63% | 5.22% |
| 3 | 0.278 | 0.253 | 6.40% | 4.47% |

On the edge probes the same direction is stronger: depth-3 route/query norm
is about `6.19%` for 300M versus `4.37%` for 500M on high-edge values. The
500M bank is not producing a proportionally stronger selected correction for
the difficult recurrent steps; its extra addresses are not becoming a
stronger depth-specific computation.

## Decision

The remaining P-007 signal is not solved by a universal correction multiplier.
The likely next intervention must improve the interface between a selected
circuit and the persistent state — for example a bounded, task-loss-aligned
state write or a deeper-task-specific state representation — while preserving
learned routing. A new route-forcing rule, static scale, or blind 700M/1B
expansion is not justified by this evidence.

## Reproduction

```powershell
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_delta2.yaml --steps 3000 --device auto --balanced-train --seed 17 --run-id ne300_delta2_s17_3000 --output results/runs --checkpoint results/checkpoints/ne300_delta2_s17_3000.pt
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_delta2.yaml --steps 3000 --device auto --balanced-train --seed 18 --run-id ne300_delta2_s18_3000 --output results/runs --checkpoint results/checkpoints/ne300_delta2_s18_3000.pt
python audit_native_depth_specialization.py --examples-per-task 32 --device auto
```

The edge probe output for the delta-2 checkpoints is stored separately as a
diagnostic only; it is not compared to a newly retrained scale-1 control in
that same invocation.

