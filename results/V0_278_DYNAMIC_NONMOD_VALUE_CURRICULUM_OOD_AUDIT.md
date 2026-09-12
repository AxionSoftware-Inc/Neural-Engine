# V0.278 — Progressive curriculum on unseen value range

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

V0.274 gave a large held-out-depth gain when the training value range was
opened as `0..7 → 0..31 → 0..95`. That screen evaluates values from the same
final range used in training. This experiment tests whether the gain reflects
a reusable arithmetic composition circuit or mainly in-range fitting.

## Design

The same 300M virtual-bank, prior-free four-digit model is trained for 5,000
steps on held-out depths 1–2 with this schedule:

```yaml
value_curriculum:
  - until_step: 1000
    value_min: 0
    value_max: 7
  - until_step: 2500
    value_min: 0
    value_max: 31
  - until_step: 5000
    value_min: 0
    value_max: 63
```

The primary evaluation uses held-out depths 3–4 and unseen values `64..95`.
For an in-range comparison, the same checkpoints are evaluated on `0..63`.
Both evaluations use 1,024 identical examples per depth and separate fixed
evaluation seed `4102`. The architecture remains unchanged at `7,483,463`
total and `2,184,304` estimated active parameters.

## Results

| Seed | Seen values 0–63 | Unseen values 64–95 | Delta | Seen CE | OOD CE | Seen d3/d4 | OOD d3/d4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 24.414% | 2.588% | −21.826 pp | 9.4201 | 27.6617 | 30.371% / 18.457% | 3.418% / 1.758% |
| 18 | 27.100% | 2.734% | −24.365 pp | 9.1299 | 26.8906 | 32.324% / 21.875% | 4.297% / 1.172% |
| **Mean** | **25.757%** | **2.661%** | **−23.096 pp** | **9.2750** | **27.2762** | **31.348% / 20.166%** | **3.857% / 1.465%** |

The in-range score is comparable to the V0.274 curriculum result, but the
unseen-value score collapses close to the wide-range control level. The OOD
evaluation still activates many factor rows (`98–121` of 154), so the result
does not look like a simple dead-router or zero-coverage failure.

## Interpretation

The positive V0.274 result is real for the distribution it trains on, but it
does not demonstrate value-compositional extrapolation. The current model
uses a learned value projection and a categorical factorized output codec;
training only on `0..63` leaves the circuit/readout behavior for the upper
range unconstrained. The evidence points to a range-dependent value-state to
output/dataflow interface, not merely insufficient total capacity.

This also means that increasing the bank to 700M or 1B before fixing the
range-transfer gate would be premature. More parameters could improve
in-range fit while leaving the OOD failure untouched.

## Decision

**V0.278 FAILS THE VALUE-EXTRAPOLATION GATE.** V0.274 remains a useful opt-in
in-range training protocol, but it is not promoted to a universal recipe or
used as evidence for scale-up. P-003/P-004 remain active.

The next experiment must keep this OOD screen and test a range-safe codec:
fixed/hybrid numeric features or an algebraic value-state bridge paired with
a numeric or otherwise extrapolating readout. A new variant is accepted only
if it improves `64..95` while retaining the in-range result and unchanged
active-compute accounting.

## Raw runs

- `results/runs/v0_278_ood64_95_paired_seed17_large.json`
- `results/runs/v0_278_ood64_95_paired_seed18_large.json`
- `results/runs/v0_278_seen0_63_paired_seed17_large.json`
- `results/runs/v0_278_seen0_63_paired_seed18_large.json`

Training configuration:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_prior_free_value_curriculum.yaml`.
