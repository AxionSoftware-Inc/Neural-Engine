# P-003 Native Factorized Long-Budget Capacity Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **capacity signal confirmed; scaling benefit remains small**

## Question

The earlier 3,000-step screens made the 500M factorized model look flat or
slightly worse than 300M. This audit repeats the matched 300M/500M ordered
factor-bank + shared factor-derived route-key configurations for 10,000 steps
from scratch. The goal is to separate insufficient optimization budget from a
real capacity ceiling.

Protocol: balanced synthetic tasks, AdamW, batch 128, numeric encoding,
adaptive halting, three recurrent steps, `active_circuits=8`, seeds 17 and 18.
The models differ only in virtual bank size (`22,800` vs `38,600`) and factor
count (`151` vs `197`).

## Results

| Scale | Mean accuracy | Mean CE | Mean dead virtual fraction | Mean time | Peak VRAM | Total params |
|---|---:|---:|---:|---:|---:|---:|
| 300M, 22,800 virtual | 77.096% | 0.66621 | 42.23% | 464.91s | 653 MB | 5.86M |
| 500M, 38,600 virtual | 77.292% | 0.65446 | 39.00% | 687.43s | 1,040 MB | 7.10M |

At equal 10,000-step training, 500M improves hard accuracy by only `+0.195 pp`
and CE by `−0.01175`. This is a real but modest capacity benefit, not the
large monotonic gain expected from a fully utilized larger model.

## Difficult-task breakdown

| Task | 300M mean | 500M mean | Change |
|---|---:|---:|---:|
| multiply | 90.43% | 82.62% | −7.81 pp |
| reverse_sum | 41.02% | 44.14% | +3.13 pp |
| chain3 | 6.84% | 6.45% | −0.39 pp |
| compose_add_mul | 32.23% | 36.72% | +4.49 pp |
| compose_if | 86.33% | 87.70% | +1.37 pp |
| state_machine | 5.27% | 6.64% | +1.37 pp |

The extra capacity helps composition and state-machine tasks modestly, but it
does not improve every task. Multiply is worse at 500M in this two-seed run,
so the aggregate gain should not be interpreted as a universal scale law.

## Routing specialization at 10,000 steps

The same 720-row specialization diagnostic was run on all four checkpoints.
Mean values across seeds are:

| Scale | Candidate pair cosine | Selected pair cosine | Unique selected | Selected bank fraction |
|---|---:|---:|---:|---:|
| 300M | 0.24459 | 0.26326 | 3,926 | 17.22% |
| 500M | 0.24523 | 0.25070 | 4,912 | 12.73% |

Both scales used all factor rows. The 500M bank has similar candidate diversity,
lower selected redundancy, and more unique selected addresses; its lower bank
fraction is the expected denominator effect from the larger bank. This rules
out the simple explanation that the larger model is entirely dead or that the
factor rows receive no gradient. The remaining issue is how much task-useful
specialization the additional virtual combinations provide.

## Decision

The 500M model should not be rejected as “not learning”; with enough training
it reaches `77.292%`, close to but above the 300M `77.096%` result. However, the
gain is too small to justify an immediate 700M/1B expansion, especially because
the 700M 1,000-step screen already showed `63.44%` virtual dead traffic. Keep
500M ordered shared-route-key as the current capacity reference and treat
10,000 steps as the minimum serious budget for this scale.

The next work should target task-useful specialization or a harder held-out
capacity benchmark, not blindly add virtual addresses. Any 700M/1B run should
wait until a new mechanism beats the 500M reference under the same 10,000-step
protocol.

## Reproduction

```powershell
python -u train.py --config configs/ne_300m_v12_factorized_shared_routekeys.yaml --steps 10000 --device cuda --balanced-train --log-every 2000 --run-id native_factorized_shared_routekeys_300m_s17_10000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_shared_routekeys.yaml --steps 10000 --device cuda --balanced-train --log-every 2000 --run-id native_factorized_shared_routekeys_300m_s18_10000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys.yaml --steps 10000 --device cuda --balanced-train --log-every 2000 --run-id native_factorized_shared_routekeys_500m_s17_10000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_shared_routekeys.yaml --steps 10000 --device cuda --balanced-train --log-every 2000 --run-id native_factorized_shared_routekeys_500m_s18_10000 --output results/runs --seed 18
```
