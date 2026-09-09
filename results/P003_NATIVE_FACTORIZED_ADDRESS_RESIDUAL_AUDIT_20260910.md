# P-003 Native Factorized Address-Residual Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a short-screen quality fix; implementation retained opt-in**

## Question

The 300M factorized-global checkpoint used every factor row, but its virtual
circuit outputs were highly correlated (`candidate_mean_pair_cosine ≈ 0.396`
versus `≈ 0.043` for the independent 300M bank). The hypothesis was that a
small address-specific residual MLP would restore distinctions between virtual
addresses without restoring a full independent bank.

The residual is a separate rank-2 circuit path per virtual address. It is added
to the factorized circuit output only for selected addresses. The existing
hierarchical global router, factorized base bank, task protocol, and active
path were unchanged.

## Protocol

- Config: `configs/ne_300m_v12_factorized_global_resid2.yaml`
- Virtual addresses: `22,800`
- Factor rows: `151`
- Address residual rank: `2`
- Steps: `1,000`
- Batch: `128`, balanced tasks, AdamW, three internal steps
- Active circuits: `8`, adaptive halting enabled
- Seeds: `17`, `18`
- Device: NVIDIA GeForce RTX 3060

The comparison is against the same 300M factorized-global 1,000-step screen:
`59.232%` mean accuracy and `1.41985` mean CE.

## Results

| Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Factorized + global, rank-2 address residual | 58.828% | 1.42390 | 56.36M | 67.83s | 1,413 MB |
| Factorized + global, no address residual | 59.232% | 1.41985 | 12.58M | 48.06s | 733 MB |

Per-seed residual values:

- Seed17: `58.594% / 1.42501 CE`, `67.74s`
- Seed18: `59.062% / 1.42280 CE`, `67.92s`

Relative to the baseline, the residual loses `0.404 pp` mean hard accuracy,
worsens mean CE by about `0.00405`, increases training time by about `41%`,
and increases peak VRAM by about `93%`. Factor-row coverage remains complete
(`151/151`); the residual did not address the underlying fragmented virtual-ID
traffic in this short screen.

## Decision

The rank-2 per-address residual is **rejected as a quality/capacity fix**. It
is retained in the code as an opt-in experimental mechanism because it is
covered by tests and may still be useful for a controlled initialization or
longer-training study, but it must not replace the retained 300M
factorized-global baseline.

The result also weakens the simple “add a unique residual to every virtual
address” explanation: adding address-local degrees of freedom without a
shared training signal creates more sparse parameters, not reliable
specialization. The next experiment should preserve the global router and add
shared, address-conditioned structure or a distillation/assignment signal,
rather than adding independent residual islands.

## Reproduction

```powershell
python -u train.py --config configs/ne_300m_v12_factorized_global_resid2.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_global_300m_resid2_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_global_resid2.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_global_300m_resid2_s18_1000 --output results/runs --seed 18
```

Artifacts:

- `results/runs/native_factorized_global_300m_resid2_s17_1000.json`
- `results/runs/native_factorized_global_300m_resid2_s18_1000.json`
- `neural_engine/circuits.py`
- `tests/test_forward.py`
