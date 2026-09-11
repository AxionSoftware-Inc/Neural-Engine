# V0.237 — operation-conditioned output adapter screen

**Date:** 2026-09-11  
**Status:** `REJECTED; MULTIPLY UNCHANGED`

## Question

V0.236 showed that the current above-range multiply path is at `0%` in both
the hard-context treatment and its control, while hard context mainly helps
subtract. This screen adds a small operation-specific low-rank adapter to the
output readout. The adapter receives the last executed primitive operation;
shorter programs reuse that operation at padded terminal steps. The bank,
router, recurrent writer, and active-8 circuit budget remain unchanged.

## Protocol

- no-interaction four-digit base-512 rank-128 control;
- operation-output adapter rank `16`, scale `1.0`;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`, compact factorized evaluator;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- primary readout: held-out multiply accuracy and CE, with add/subtract as
  controls.

The adapter is retained only if it produces non-zero and repeatable multiply
accuracy without damaging add/subtract or materially inflating the active
path. This is an opt-in diagnostic, not a default change.

## Results

The two fresh runs completed. Against the matched no-interaction rank-128
reference, the aggregate held-out result changes only from `58.2031%` to
`58.6914%` (`+0.4883 pp`), depth-3 from `63.8672%` to `64.6484%`
(`+0.7813 pp`), and depth-4 from `52.5391%` to `52.7344%`
(`+0.1953 pp`). Mean CE regresses from `10.892668` to `12.438559`
(`+1.545890`). The adapter adds `38,016` total/active-estimate parameters
(`7,477,191 → 7,515,207` total; `2,178,032 → 2,216,048` active estimate).

| operation | control held-out | adapter held-out | delta |
|---|---:|---:|---:|
| add | 100.00% | 100.00% | 0.00 pp |
| subtract | 55.7617% | 58.6914% | +2.9297 pp |
| multiply | 0.00% | 0.00% | 0.00 pp |

On the operation-specific diagnostic, subtract depth-4 improves from
`27.3438%` to `34.1797%`, but subtract depth-3 falls from `84.1797%` to
`83.2031%`; multiply remains exactly `0%` at both held-out depths in both
arms. The adapter therefore does not address the primary ceiling and its
small aggregate gain is not worth the CE regression.

## Decision

**REJECT FOR ADOPTION.** Operation-conditioned readout is not enough to make
the unseen multiply path work. Keep the code as an opt-in diagnostic, leave
the default unchanged, and do not scale capacity. The next experiment should
preserve the exact algebraic value packet directly into the output codec,
rather than only conditioning a learned readout on the operation ID.
