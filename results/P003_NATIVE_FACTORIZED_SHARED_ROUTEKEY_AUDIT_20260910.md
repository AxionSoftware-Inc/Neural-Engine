# P-003 Native Factorized Shared-Route-Key Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **mixed; combined ordered variant retained opt-in, route-key-only fix not accepted**

## Question

The factorized circuit bank reuses factor rows, but the ordinary global router
still stores an independent key for every virtual address. That mismatch can
fragment traffic and give the router geometry no relation to the circuit
representation.

This experiment adds a global hierarchical router mode whose candidate key for
virtual address `(i, j)` is generated from two reusable factor keys. The global
tree, local candidate pool, and hard top-k execution remain unchanged. This is
different from the earlier `FactorizedRouter`, which narrows retrieval by
first selecting a small factor pool.

## Important controls

Two factors must be separated:

1. **Route-key-only control:** unordered factor bank plus shared factor-derived
   route keys. This isolates the router-key representation.
2. **Combined candidate:** ordered factor bank plus shared factor-derived route
   keys. This tests the best interaction of slot asymmetry and aligned routing.

## Results

### 500M, 1,000-step screen

| Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Shared factor bank + ordinary global keys | 58.503% | 1.42049 | 19.27M | 72.30s | 1,225 MB |
| Shared factor bank + shared route keys (unordered) | 58.841% | 1.42483 | 4.52M | 67.33s | 999 MB |
| Ordered bank + shared route keys | 59.258% | 1.40675 | 7.09M | 69.14s | 1,039 MB |

The unordered route-key-only arm improves hard accuracy by `+0.339 pp` but
worsens CE by `0.00434`. The ordered combined arm improves accuracy by
`+0.755 pp` and CE by `0.01374`, but it changes both the bank slot structure
and the router keys, so it is not a pure route-key attribution.

### 500M, 3,000-step continuation

| Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Shared factor bank + ordinary global keys | 68.281% | 0.99648 | 19.27M | 215.75s | 1,225 MB |
| Shared factor bank + shared route keys (unordered) | 68.138% | 0.99588 | 4.52M | 200.63s | 999 MB |
| Ordered bank + shared route keys | 68.594% | 0.98665 | 7.09M | 206.30s | 1,039 MB |

At 3,000 steps, route-key-only is `−0.143 pp` below the ordinary-key baseline
despite slightly better CE. The combined ordered arm is `+0.313 pp` above
that baseline and uses `63%` fewer stored parameters and `15%` less VRAM.

### 300M and 700M combined ordered screens

| Scale | Mean accuracy | Mean CE | Mean virtual dead fraction |
|---:|---:|---:|---:|
| 300M combined, 3,000 steps | 68.451% | 0.99639 | 40.17% |
| 500M combined, 3,000 steps | 68.594% | 0.98665 | 44.01% |
| 700M combined, 1,000 steps | 58.815% | 1.41435 | 63.44% |

The combined 300→500 improvement is only `+0.143 pp` at equal 3,000-step
protocols, and the 700M short screen falls sharply while virtual dead traffic
rises. This does not justify a 700M long run or a 1B expansion.

## Decision

Shared factor-derived route keys are a useful compression/alignment mechanism,
but **route-key-only is not a reliable quality fix**. The ordered+route-key
combination is retained as the strongest 500M opt-in candidate because it beats
the ordinary 500M baseline by `+0.313 pp` hard accuracy at 3,000 steps, but the
gain is not monotonic through 700M and remains below the confidence needed for
default promotion.

The main unresolved issue is virtual-address fragmentation at larger banks.
The next work should improve factor-aware candidate assignment or train a
shared route objective; blindly increasing address count is rejected.

## Reproduction

```powershell
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_unordered.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_500m_unordered_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_unordered.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_500m_unordered_s18_3000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_500m_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_500m_s18_3000 --output results/runs --seed 18
python -u train.py --config configs/ne_700m_v12_factorized_shared_routekeys_ordered.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_700m_ordered_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_700m_v12_factorized_shared_routekeys_ordered.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_700m_ordered_s18_1000 --output results/runs --seed 18
```
