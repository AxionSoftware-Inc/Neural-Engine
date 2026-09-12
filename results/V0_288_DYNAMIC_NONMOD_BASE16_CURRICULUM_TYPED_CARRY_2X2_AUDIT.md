# V0.288 — Base-16 curriculum × typed-carry 2×2 screen

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `RETAINED AS DIAGNOSTIC; REJECTED FOR QUALITY ADOPTION`

## Question

V0.274 showed a large benefit from progressive value difficulty on the earlier
four-digit base-512 output path. V0.286/V0.287 showed that the eight-digit
typed-carry path has a useful local signal but weak deep transfer. V0.288
separates these effects with a matched 2×2:

1. eight-digit base-16 control, full-range training;
2. the same control with progressive value curriculum;
3. typed carry, full-range training;
4. typed carry with the same curriculum.

## Protocol

- 300M virtual-bank factorized Neural Engine, active 8 circuits;
- values `0..95`, train depths `1..2`, held-out depths `3..4`;
- eight base-16 output digits, rank 128, interaction rank 16;
- typed treatment: eight 16D slots, shared low-to-high learned carry chain,
  intermediate typed-digit contract loss `0.5`;
- curriculum: `0..7` through step 1,000, `0..31` through step 2,500,
  `0..95` through step 5,000;
- 5,000 fresh optimization steps, batch `128`, 1,024 examples per held-out
  depth, CUDA, seeds `17/18`;
- no teacher forcing, exact arithmetic packet, modular prior, or forced active
  path in any arm.

## Results

| Arm | Seed | Held-out acc | CE | Depth 3 | Depth 4 | Train sec |
|---|---:|---:|---:|---:|---:|---:|
| Base-16 control, plain | 17 | 22.4609% | 12.2262 | 28.1250% | 16.7969% | 774.4 |
| Base-16 control, plain | 18 | 19.0918% | 12.9883 | 24.7070% | 13.4766% | 728.4 |
| Base-16 control, curriculum | 17 | 18.6035% | 12.7682 | 23.6328% | 13.5742% | 783.5 |
| Base-16 control, curriculum | 18 | 17.8223% | 12.0549 | 23.4375% | 12.2070% | 780.5 |
| Typed carry, plain | 17 | 18.6523% | 12.1728 | 24.3164% | 12.9883% | 952.6 |
| Typed carry, plain | 18 | 17.3828% | 11.9605 | 23.1445% | 11.6211% | 951.1 |
| Typed carry, curriculum | 17 | 20.7031% | 10.5145 | 26.7578% | 14.6484% | 968.1 |
| Typed carry, curriculum | 18 | 19.0918% | 11.1295 | 25.8789% | 12.3047% | 938.5 |

Two-seed means:

| Arm | Held-out acc | CE | Depth 3 | Depth 4 | Train sec |
|---|---:|---:|---:|---:|---:|
| Base-16 control, plain | 20.7764% | 12.6073 | 26.4160% | 15.1367% | 751.4 |
| Base-16 control, curriculum | 18.2129% | 12.4115 | 23.5352% | 12.8906% | 782.0 |
| Typed carry, plain | 18.0176% | 12.0666 | 23.7305% | 12.3047% | 951.9 |
| Typed carry, curriculum | 19.8975% | 10.8220 | 26.3184% | 13.4766% | 953.3 |

Curriculum changes the typed-carry arm by `+1.8799 pp` accuracy, `−1.2446`
CE, `+2.5879 pp` depth-3 accuracy, and `+1.1719 pp` depth-4 accuracy. It does
not beat the plain control: typed-curriculum is `−0.8789 pp` overall and
`−1.6601 pp` at depth 4, although its CE is `1.7853` lower. The curriculum
also hurts the control arm by `−2.5635 pp` accuracy, so the earlier curriculum
gain is not codec-independent.

The typed path estimates `2,047,296` active parameters versus `1,989,312` for
the control (`+57,984`) and takes about `26.9%` more training time. Inference
still executes only the selected 8 circuits; curriculum changes training data,
not active inference compute.

## Interpretation and decision

The result is useful, but not a quality breakthrough. Progressive difficulty
helps the typed carry representation relative to its own plain baseline, and
it substantially improves CE, yet the simplest base-16 control remains the
best hard-selection model. This means the curriculum can improve optimization
of a weak representation, but it does not prove that the typed carry transition
is a better reusable circuit.

**Decision:** retain both configurations as opt-in diagnostics; do not replace
the default, do not increase typed-carry capacity, and do not scale this result
to 700M/1B. P-003/P-004 remain active. The next Native experiment should target
the transition/dataflow contract directly and must beat the plain base-16
control on hard accuracy, not only CE.

## Raw reports

- `results/runs/v0_288_base16_curriculum_2x2_control_plain_seed17_5000.json`
- `results/runs/v0_288_base16_curriculum_2x2_control_plain_seed18_5000.json`
- `results/runs/v0_288_base16_curriculum_2x2_control_curriculum_seed17_5000.json`
- `results/runs/v0_288_base16_curriculum_2x2_control_curriculum_seed18_5000.json`
- `results/runs/v0_288_base16_curriculum_2x2_typed_plain_seed17_5000.json`
- `results/runs/v0_288_base16_curriculum_2x2_typed_plain_seed18_5000.json`
- `results/runs/v0_288_base16_curriculum_2x2_typed_curriculum_seed17_5000.json`
- `results/runs/v0_288_base16_curriculum_2x2_typed_curriculum_seed18_5000.json`

