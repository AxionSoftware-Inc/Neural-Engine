# P-007 native gated state-write audit

Date: 2026-09-10  
Branch: `exp/track-runtime`

## Hypothesis

The native model uses a persistent GRU state. A selected circuit correction is
fed into that update, but the update may overwrite useful intermediate state
on multi-step tasks. `memory_write_mode=gated` adds a learned write gate over
the proposed GRU state, initialized close to the original path. It does not
force routes or change the active circuit budget.

## Matched screen

300M ordered shared-route-key bank, balanced training, 3,000 steps, seed17/18:

| Arm | Mean accuracy | Mean CE | Depth-2/3 mean |
|---|---:|---:|---:|
| shared route-key baseline | 68.451% | 0.99639 | 33.366% |
| gated state write | 67.969% | 0.99445 | 33.203% |
| delta | **−0.482 pp** | **−0.00194** | **−0.163 pp** |

The gated arm adds 295,296 parameters and raises peak VRAM from 653 MB to
663 MB. It does not improve the difficult tasks; it slightly lowers their
mean accuracy despite a small CE improvement.

## Decision

**REJECTED as a quality fix.** The state-write gate is retained as an opt-in
control, but it is not promoted and is not used as the next scaling path. The
result strengthens P-005/P-007: the problem is not solved by a generic state
preservation gate or a universal correction amplitude.

## Reproduction

```powershell
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_memory_write.yaml --steps 3000 --device auto --balanced-train --seed 17 --run-id ne300_memory_write_s17_3000 --output results/runs --checkpoint results/checkpoints/ne300_memory_write_s17_3000.pt
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_memory_write.yaml --steps 3000 --device auto --balanced-train --seed 18 --run-id ne300_memory_write_s18_3000 --output results/runs --checkpoint results/checkpoints/ne300_memory_write_s18_3000.pt
```

