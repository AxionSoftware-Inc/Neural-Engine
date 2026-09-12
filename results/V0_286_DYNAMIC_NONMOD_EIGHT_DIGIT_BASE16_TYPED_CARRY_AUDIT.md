# V0.286 — eight base-16 digits with a learned carry-chain state

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `RETAINED AS DIAGNOSTIC; REJECTED FOR QUALITY ADOPTION`

## Question

The leading four-digit base-512 codec still makes each local digit decision
large. V0.286 tests whether a learned carry-chain becomes easier to train when
the same `2^33` class space is represented by eight base-16 digits. The
treatment adds a typed eight-slot recurrent state and a least-significant to
most-significant learned carry packet. The output also uses eight base-16
heads. No exact arithmetic table or integer packet is exposed.

## Protocol

- 300M virtual-bank Neural Engine configuration;
- operands `0..95`, train depths `1..2`, held-out depths `3..4`;
- output: eight base-16 digits, shared rank `128`, interaction rank `16`;
- treatment: eight base-16 typed slots, dimension `16` each, learned carry chain;
- control: same eight-digit output, no typed state;
- 2,000 fresh steps, batch `128`, CUDA;
- seeds `17` and `18`, 1,024 identical examples per held-out depth;
- same active-8 factorized circuit route in both arms.

## Results

| Arm | Seed | Held-out accuracy | Depth 3 | Depth 4 | CE | Train accuracy | Total params | Active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 17 | 3.174% | 3.711% | 2.637% | 12.1439 | 28.418% | 7,288,471 | 1,989,312 |
| Typed carry | 17 | 3.125% | 3.906% | 2.344% | 11.7885 | 26.025% | 7,346,455 | 2,047,296 |
| Control | 18 | 2.686% | 3.418% | 1.953% | 12.4945 | 23.730% | 7,288,471 | 1,989,312 |
| Typed carry | 18 | 3.857% | 5.176% | 2.539% | 12.1981 | 22.119% | 7,346,455 | 2,047,296 |
| **Control mean** | — | **2.930%** | **3.564%** | **2.295%** | **12.3192** | **26.074%** | — | — |
| **Typed carry mean** | — | **3.491%** | **4.541%** | **2.441%** | **11.9933** | **24.072%** | — | — |

Treatment minus control is `+0.562 pp` overall, `+0.977 pp` at depth 3, and
`+0.146 pp` at depth 4, with CE improving by `−0.3259`. The direction is not
consistent by seed: seed17 is slightly worse overall, while seed18 improves.
The gain is below the project's `+2 pp` adoption gate, and train accuracy is
not improved. The typed state adds `57,984` total/active-estimate parameters
and does not increase the number of selected circuits.

## Interpretation

Smaller digit alphabets and a local learned carry packet provide a modest CE
and depth-3 signal, but they do not recover compositional quality. This rules
out “large base-512 digit heads alone” as the only explanation, while also
showing that digit granularity is not sufficient without a stronger learned
operation transition. The result is not evidence for router starvation or for
700M/1B scaling.

## Decision

**Do not adopt V0.286 as the quality path.** Keep the configuration and
checkpoints as an opt-in diagnostic. P-003/P-004 remain active; the next
architecture experiment should target an explicit reusable state transition
or teacher-forced transition distillation, not another output-base split.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_rank128.yaml`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128.yaml`
- corresponding reports under `results/runs/`
- corresponding checkpoints under `results/checkpoints/`
