# P-003 native 500M routing-capacity clamp audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Base recipe: 500M ordered shared-route-key bank + rank-8 step adapter + 25%
low/high-edge training mix

## Question

The full 500M router exposes 38,600 virtual circuits and routing depth 6,
while the 300M reference exposes 22,800 and depth 5. The full 500M mixed run
had more dead traffic and worse ordinary quality. This control keeps the 500M
parameter bank but limits the router to the 300M reachable prefix:
`routing_capacity=22800`, `routing_depth=5`.

## Protocol

- 10,000 steps, balanced task batches, batch 128, AdamW;
- seeds 17 and 18;
- identical 24-batch evaluator for uniform, combination-holdout, low-edge and
  high-edge probes;
- all models use the same 25% two-edge training mix.

## Full 500M versus clamped 500M

| Probe | Full 500M | Clamped 500M | Clamp delta | Hard-task delta |
|---|---:|---:|---:|---:|
| uniform all | 76.736% | 77.231% | **+0.495 pp** | **+1.259 pp** |
| combination holdout | 77.678% | 78.186% | **+0.508 pp** | **+1.172 pp** |
| low edge `[0,7]` | 96.011% | 95.629% | −0.382 pp | −0.836 pp |
| high edge `[56,63]` | 95.482% | 95.764% | **+0.282 pp** | **+0.803 pp** |

The clamp improves ordinary and high-edge quality compared with full 500M.
The 500M clamped model still trails the 300M mixed reference by `0.634 pp`
on uniform accuracy and `0.933 pp` on uniform hard-task mean. Low-edge is
also `0.095 pp` below 300M, while high-edge is `+1.202 pp` above it.

## Diagnosis

This is evidence for route fragmentation as a real component of the scaling
problem: reducing the reachable prefix and tree depth gives a measurable
ordinary-quality recovery without changing active circuit count. It is not
evidence that 500M has become a better model; extra stored rows still do not
produce monotonic quality scaling.

## Decision

**PROMISING DIAGNOSTIC; not default.** Keep the clamp as a controlled
capacity-utilization baseline. Do not promote 500M or proceed to 700M/1B yet.
The next architecture work should make new capacity earn active credit (for
example inherited/shared route geometry or a learned capacity allocation),
then re-run the same two-edge recipe and edge gates.

## Raw evidence

- `results/diagnostic_native_two_edge_mix_clamp_300m_500m_10000.json`
- `results/diagnostic_native_two_edge_mix_capacity_clamp_500m_3000.json`
- `configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_capacity_clamp.yaml`
