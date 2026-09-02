# V0.51 dynamic proposal screen: proposals 9–15

Status: **screen complete; no proposal produced a reliable large jump**

These screens use the same attention-free Dynamic Register Machine and the
same OOD gate throughout: 20M virtual capacity, 3,000 steps, batch 512,
training depths 1–4 on values 0–31, evaluation depths 5–6 on values 32–63.
The baseline is 34.77% at seed 17 unless otherwise noted.

## Results

| Proposal / variant | OOD accuracy | Training seconds | Decision |
|---|---:|---:|---|
| Baseline, serial, rank 16, exploration 5% | 34.77% | 341.4 | reference |
| 9. Route audit | — | — | diagnostic only; reuse confirmed |
| 10. State 256 | 32.91% | 304.0 | reject |
| 10. State 512 | 32.91% | 391.6 | reject |
| 11. Input reinjection 0.25 | 35.55% | 341.2 | small, unconfirmed |
| 12. Generic gated write | 34.96% | 349.9 | reject |
| 13. Circuit rank 32 | 33.89% | 457.3 | reject |
| 14. Parallel circuit mix | 35.16% | 285.5 | quality neutral; speed candidate |
| 15. Exploration 0% | 34.86% | 342.8 | neutral |
| 15. Exploration 20%, seed 17 | 36.13% | 344.9 | not sufficient alone |
| 15. Exploration 20%, seed 18 | 31.64% | 344.4 | fails repeatability |

The exploration-20% two-seed mean is **33.89%**, below the baseline. The
parallel variant is about 16% faster in this implementation, but its +0.39
point quality difference is not a meaningful OOD improvement.

## Interpretation

The screen rules out several attractive but broad fixes:

- changing working-state width from 256 to 512 does not repair value
  extrapolation;
- a generic memory gate does not repair it;
- doubling low-rank circuit rank does not repair it; and
- more random routing does not repair it reliably.

Input reinjection is the only quality-positive single run, but the gain is
less than one point and has not passed a second seed. It is not promoted to
300M. Parallel mix remains useful only as a possible systems optimization,
not as a quality solution.

Together with V0.50, the evidence is now strong that the failure is specific
to value-independent `add/subtract` extrapolation. The next architecture
work should target the value/operation representation or an exact modular
composition interface, while preserving the recurrent register and sparse
router. 500M/700M scaling is still gated on this test.

Reproduction run IDs:

```text
ne_dynamic_20m_state256_unseen_values_3000
ne_dynamic_20m_state512_unseen_values_3000
ne_dynamic_20m_reinject_unseen_values_3000
ne_dynamic_20m_write_gate_unseen_values_3000
ne_dynamic_20m_rank32_unseen_values_3000
ne_dynamic_20m_parallel_unseen_values_3000
ne_dynamic_20m_explore0_unseen_values_3000
ne_dynamic_20m_explore20_unseen_values_3000
ne_dynamic_20m_explore20_unseen_values_seed18_3000
```
