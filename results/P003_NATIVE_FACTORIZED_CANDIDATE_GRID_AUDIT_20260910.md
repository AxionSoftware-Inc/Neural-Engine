# P-003 Native Factorized Candidate-Grid Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a quality fix; retained opt-in for diagnostics**

## Question

The shared factor-derived route-key experiment reduced circuit similarity, but
the ordinary flat candidate window still advances mostly along one factor
coordinate. This audit tests a Cartesian `4x8` candidate pool around the routed
factor address. A second arm adds a parameter-free factor-pair interaction score
to the candidate logits so that hard top-k selection can prefer complementary
pairs.

Both arms keep the ordered factor bank, shared factor-derived route keys,
`candidate_pool=32`, `active_circuits=8`, and the same 500M protocol. Only
candidate construction or candidate scoring changes; execution remains sparse.

## Results

### 1,000-step screen

| Arm | Mean accuracy | Mean CE | Peak VRAM |
|---|---:|---:|---:|
| Ordered bank + shared route keys, flat pool | 59.258% | 1.40675 | 1,039 MB |
| Ordered bank + shared route keys, factor-grid pool | 59.440% | 1.40610 | 1,039 MB |
| Factor-grid + pair interaction score (`scale=1`) | 59.232% | 1.40638 | 1,075 MB |

The grid screen is a small positive signal, but the pair-interaction arm does
not improve it. The screen therefore does not justify a long run for the pair
score.

### 3,000-step continuation

| Arm | Mean accuracy | Mean CE | Mean virtual dead fraction |
|---|---:|---:|---:|
| Shared factor bank + ordinary global keys | 68.281% | 0.99648 | 44.01% |
| Ordered bank + shared route keys, flat pool | 68.594% | 0.98665 | 44.01% |
| Ordered bank + shared route keys, factor-grid pool | 68.047% | 0.99621 | 40.62% |

The factor-grid arm loses the short-screen gain and falls `0.547 pp` below
the ordered route-key arm. Its lower dead fraction is not sufficient: the
selected circuits become more redundant.

## Structural diagnostic

The seed17 factor-grid checkpoint was evaluated on the same 720-row diagnostic
set as the matched flat-pool checkpoint:

| Arm | Candidate pair cosine | Selected pair cosine | Selected unique | Selected bank fraction |
|---|---:|---:|---:|---:|
| Ordered route keys, flat pool | 0.26475 | 0.27034 | 4,742 | 12.28% |
| Ordered route keys, factor-grid pool | 0.19770 | 0.32222 | 4,850 | 12.56% |

The grid exposes more diverse candidates, but hard top-k chooses a more
correlated subset. This is evidence that the unresolved bottleneck is the
selection objective or credit assignment, not simply the number of factor
coordinates present in the candidate pool.

## Decision

Reject factor-grid and factor-pair interaction as default quality changes.
Keep both as opt-in research controls. Do not expand this arm to 700M or 1B.
The next architecture-level work should address complementary subset
selection with a directly measured final-output objective, or revisit the
factor-address representation itself. Repeating larger screens without that
change is unlikely to resolve the scaling failure.

## Reproduction

```powershell
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_grid.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_grid_500m_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_grid.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_grid_500m_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_grid.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_grid_500m_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_grid.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_grid_500m_s18_3000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_grid_pair.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_grid_pair_500m_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_grid_pair.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_grid_pair_500m_s18_1000 --output results/runs --seed 18
```
