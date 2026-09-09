# P-003 Native Factorized Ordered-Slots Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **promising at 300M; matched 500M gate failed; not default**

## Question

The original factorized bank used one reusable factor table for both positions
of a virtual address `(i, j)`. That makes the two slots structurally symmetric
and can make different virtual circuits too similar. This experiment keeps the
same factorized representation and the existing hierarchical global router,
but uses two separate reusable tables: one for the first slot and one for the
second slot.

This is shared slot structure, not an independent parameter island per virtual
address.

## Protocol

- Config: `configs/ne_300m_v12_factorized_global_ordered.yaml`
- Virtual addresses: `22,800`
- Factor rows per slot: `151`
- Ordered factor slots: enabled
- Steps: `1,000` screen and `3,000` continuation
- Batch: `128`, balanced tasks, AdamW, three internal steps
- Active circuits: `8`, adaptive halting enabled
- Seeds: `17`, `18`
- Device: NVIDIA GeForce RTX 3060

Baseline is the same 300M factorized-global architecture with one shared factor
table for both slots.

## Results

| Steps | Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | shared factor table | 59.232% | 1.41985 | 12.58M | 48.06s | 733 MB |
| 1,000 | ordered slot tables | 59.036% | 1.41467 | 14.49M | 49.31s | 754 MB |
| 3,000 | shared factor table | 68.242% | 0.99648 | 12.58M | 142.74s | 733 MB |
| 3,000 | ordered slot tables | 68.451% | 0.98753 | 14.49M | 146.89s | 754 MB |

Matched 500M scale gate (same `d_model=384` as the shared baseline):

| Steps | Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 3,000 | shared factor table | 68.281% | 0.99648 | 19.27M | 215.75s | 1,225 MB |
| 3,000 | ordered slot tables | 68.073% | 1.01038 | 21.76M | 223.41s | 1,263 MB |

Per-seed 3,000-step ordered values:

- Seed17: `68.177% / 0.99729 CE`, `146.83s`
- Seed18: `68.724% / 0.97778 CE`, `146.95s`

Both seeds beat the corresponding global-router baseline in hard accuracy.
The mean gain is `+0.208 pp`; mean CE improves by `0.00895`. The cost is
about `15%` more stored factor-bank parameters, `3%` more training time, and
`21 MB` more peak VRAM. All `151/151` factor rows are used. Virtual-address
dead fraction remains high, so this result does not prove that fragmentation
has been solved.

The first 500M ordered run used `d_model/state_dim=512`, while the shared 500M
baseline uses `384`; its `69.818%` result is therefore a confounded width
comparison and is not used as evidence for ordered slots. The matched 500M
run removes that confound and regresses by `0.208 pp` accuracy with worse CE.

## Decision

Ordered factor slots remain a useful 300M opt-in candidate, but the matched
500M gate failed. Do not promote it to default or expand it to 700M/1B as a
solution. The next experiment should add shared query-conditioned composition
on top of the ordered tables, not more width or more address-local parameters.

## Reproduction

```powershell
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_ordered_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_ordered_s18_3000 --output results/runs --seed 18
```

Artifacts:

- `results/runs/native_factorized_global_300m_ordered_s17_3000.json`
- `results/runs/native_factorized_global_300m_ordered_s18_3000.json`
- `configs/ne_300m_v12_factorized_global_ordered.yaml`
