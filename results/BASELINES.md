# Frozen comparison baselines

These are the numbers to use when evaluating new work. They come from the
audits linked below and are not interchangeable benchmarks.

| Track | Reference | Quality | Active compute | Runtime | Decision |
|---|---|---|---|---|---|
| Native | NE-V0.12 | `71.98%` validation accuracy; dense reference `51.02%` | `1.977M / 20.247M` unique active params | `8.322 ms`, `15,382/s` at batch 128 | Frozen baseline |
| Qwen sparse | K=6 | CE delta `+0.01141/+0.01837` across two seeds | `75%` of teacher FFN experts | comparable trained timing `0.991x/1.004x` | Quality reference |
| Qwen sparse | K=5 | CE delta `+0.03881/+0.04036` across two seeds | `62.5%` of teacher FFN experts | `1.134x/1.129x` in the comparable smoke | Lower-budget reference |
| Qwen sparse | K=4 learned | CE delta `+0.06462/+0.06165` | `50%` of teacher FFN experts | not an adopted baseline | Open routing failure |
| Qwen sparse | K=4 paired oracle | CE delta `+0.01607/+0.01227` | `50%` of teacher FFN experts | oracle only | Diagnostic upper bound |

## Scope warning

Native numbers are from the repository's synthetic numeric/composition task.
Qwen numbers are teacher-relative FFN reconstruction/quality controls. A new
experiment must compare only with the baseline from its own track.

## Source reports

- Native: `CHECKPOINTED_INFERENCE.md`
- Qwen protocol: `V0_174_CORRECTED_ROUTING_AND_EXPERT_AUDIT.md`
- Qwen dispatch: `V0_193_QWEN_CORRECTION_DISPATCH_AUDIT.md`
- Qwen K=4 rejection: `V0_194_QWEN_K4_ON_POLICY_PAIRWISE_AUDIT.md`

