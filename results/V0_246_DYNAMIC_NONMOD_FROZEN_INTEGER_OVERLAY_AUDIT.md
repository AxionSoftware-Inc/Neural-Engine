# V0.246 — frozen multiply integer-output overlay

**Date:** 2026-09-11  
**Status:** `RETAINED AS STRONG OPT-IN SIGNAL; NOT DEFAULT`

## Question

Can the exact integer packet improve the difficult multiply path without
disturbing the already-good add/subtract paths? V0.245 retrained the whole
model and improved multiply while collapsing subtract. V0.246 therefore loads
the matched V0.240 checkpoint, freezes every existing parameter, and trains
only a new exact-integer decoder plus a separate multiply-only factorized digit
head.

## Protocol

- matched V0.240 hard-context checkpoints, seeds `17` and `18`;
- operands `0..95`, train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, 5,000 overlay steps, batch `128`;
- only `337,474` new overlay parameters are trainable;
- base model, router, circuit bank, learned add/subtract output head, and all
  other parameters are frozen;
- taskwise evaluation uses the same `256` examples per depth as V0.240.

## Results

| Held-out metric | V0.240 hard context | V0.246 frozen overlay | Delta |
|---|---:|---:|---:|
| All operations | 82.7148% | 83.6914% | +0.9766 pp |
| Depth 3, all operations | 86.5234% | 87.3047% | +0.7813 pp |
| Depth 4, all operations | 78.9063% | 80.0781% | +1.1719 pp |
| Add | 100.0000% | 100.0000% | +0.0000 pp |
| Subtract | 99.7070% | 99.7070% | +0.0000 pp |
| Multiply | 15.4297% | 23.1445% | **+7.7148 pp** |
| Multiply, depth 3 | 23.0469% | 36.9141% | **+13.8672 pp** |
| Multiply, depth 4 | 7.8125% | 9.3750% | +1.5625 pp |

The multiply result by seed is `22.6563%/23.6328%` for V0.246 versus
`15.2344%/15.6250%` for V0.240. Add and subtract are exactly unchanged in this
matched taskwise evaluation, as expected from freezing their path.

## Interpretation

This is the first clean positive result for the exact integer packet under the
current architecture: it improves the hard multiply task while preserving the
strong paths. The gain is not a general capacity result and does not justify
700M/1B scaling by itself. The added head is operation-specific and increases
the full model from `7,500,743` to `7,838,217` parameters, although only
`337,474` parameters are trained in the overlay.

The earlier V0.245 end-to-end configuration remains rejected. V0.246 is kept
as an opt-in branch/ablation. The next experiment should test lower-rank or
shared versions of this overlay to see whether the same gain survives with a
smaller active/training budget.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integeroverlay.yaml`
- `train_integer_output_overlay.py`
- `results/runs/v0_246_frozen_integeroverlay_seed17_5000.json`
- `results/runs/v0_246_frozen_integeroverlay_seed18_5000.json`
- `results/runs/v0_246_frozen_integeroverlay_taskwise_seed17.json`
- `results/runs/v0_246_frozen_integeroverlay_taskwise_seed18.json`
- base comparison: `results/runs/v0_240_wide_range_hardcontext_taskwise_seed17.json`
  and `...seed18.json`
