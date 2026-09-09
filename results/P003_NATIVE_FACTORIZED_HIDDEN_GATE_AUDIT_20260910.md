# P-003 Native Factorized Hidden-Gate Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a scaling fix; retained opt-in**

## Question

The factorized bank has only additive factor paths. This experiment adds a
small reusable gate per factor row over the rank channels. For address `(i,j)`
the additive hidden activation is modulated by a factor-derived gate from
`gate_i + gate_j`. The gate table has only `2 * factor_count * rank` entries
for ordered slots; it is initialized at zero so the initial forward path is
unchanged.

## Results

The protocol is the 500M/38,600-address ordered bank with shared factor-derived
global route keys, balanced batches, AdamW, and seeds 17/18.

| Arm | Steps | Mean accuracy | Mean CE | Mean dead fraction | Mean time | Peak VRAM | Total params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordered shared route keys, no gate | 1,000 | 59.258% | 1.40675 | 49.07% | 69.14s | 1,039 MB | 7,091,457 |
| Hidden gate, scale=1 | 1,000 | 59.410% | 1.40286 | 50.63% | 70.37s | 1,040 MB | 7,097,761 |
| Ordered shared route keys, no gate | 3,000 | 68.594% | 0.98665 | 44.01% | 206.30s | 1,039 MB | 7,091,457 |
| Hidden gate, scale=1 | 3,000 | 68.060% | 0.99790 | 44.00% | 208.95s | 1,040 MB | 7,097,761 |

The all-seed short-screen gain of `+0.152 pp` disappears at 3,000 steps. The
long run is `−0.534 pp` below the route-key arm and `−0.221 pp` below the
ordinary global-key 500M baseline. Memory and stored-parameter costs are
almost unchanged; the failure is quality, not efficiency.

## Decision

Reject hidden factor gates as the current capacity solution. The code remains
an opt-in control because it is a cheap reusable-composition mechanism, but it
does not make quality scale with the virtual-address count. The unresolved
bottleneck remains joint route selection, factor-combination specialization,
and final-loss credit assignment.

## Reproduction

```powershell
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_gate.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_hidden_gate_500m_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_gate.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_hidden_gate_500m_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_gate.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_hidden_gate_500m_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_gate.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_hidden_gate_500m_s18_3000 --output results/runs --seed 18
```
