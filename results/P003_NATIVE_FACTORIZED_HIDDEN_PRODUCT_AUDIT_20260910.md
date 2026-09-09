# P-003 Native Factorized Hidden-Product Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a quality/efficiency fix; retained opt-in**

## Question

The additive factorized bank represents each virtual circuit as a weighted sum
of two factor paths. This may limit combination-specific capacity even when the
router finds a good address. The tested extension adds a parameter-free hidden
interaction:

```text
GELU(state @ down_first) * GELU(state @ down_second)
```

The interaction is projected through the average of the two existing up
matrices. It changes the factor composition, not the router or recurrent state
update, and does not add a large per-address parameter table.

## Results

All runs use the ordered factor bank and shared factor-derived global keys on
the 500M/38,600-address configuration, with balanced batches, seeds 17 and
18, and the same protocol as the route-key audit.

| Arm | Steps | Mean accuracy | Mean CE | Mean dead fraction | Time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Ordered bank + shared route keys | 1,000 | 59.258% | 1.40675 | 49.07% | 69.14s | 1,039 MB |
| Hidden-product interaction, scale=1 | 1,000 | 59.766% | 1.40634 | 52.62% | 89.62s | 1,493 MB |
| Ordered bank + shared route keys | 3,000 | 68.594% | 0.98665 | 44.01% | 206.30s | 1,039 MB |
| Hidden-product interaction, scale=1 | 3,000 | 68.307% | 0.99133 | 44.28% | 268.34s | 1,493 MB |

The short screen's `+0.508 pp` mean gain does not survive the longer run. At
3,000 steps the hidden-product arm is `−0.287 pp` below the route-key arm and
uses about `30%` more training time and `44%` more VRAM. It has the same stored
parameter count because the interaction reuses existing matrices.

## Structural diagnostic

On the same 720-row seed17 specialization set, the hidden-product checkpoint
had candidate/selected pair cosine `0.27228/0.27389`, versus
`0.26475/0.27034` for the flat route-key checkpoint. Selected unique circuits
were `4,894` versus `4,742`. Thus the interaction changes circuit outputs but
does not reduce the measured routing fragmentation or selected redundancy.

## Decision

Reject hidden-product composition as the current scaling fix. Keep the code as
an opt-in mathematical control because it is a valid nonlinear composition
hypothesis with no new large parameter bank, but do not promote it or expand it
to 700M/1B. The main open issue remains how reusable factor combinations acquire
distinct, task-useful specialization under hard routing.

## Reproduction

```powershell
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_product.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_hidden_product_500m_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_product.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_shared_routekeys_hidden_product_500m_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_product.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_hidden_product_500m_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys_hidden_product.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_shared_routekeys_hidden_product_500m_s18_3000 --output results/runs --seed 18
```
