# V0.37 factorized virtual capacity

Status: **positive through 300M; 500M regression remains open**

## Change

V0.37 keeps the V0.36 multiplicative pair interaction and replaces the
independent circuit bank with a factorized virtual bank. Each virtual circuit
ID is composed from two reusable factor IDs. The low-rank down/up matrices and
bias are weighted sums of the two factor rows; a tiny per-address mix code
preserves address-specific variation.

```text
virtual route ID
  -> factor A + factor B
  -> composed low-rank circuit
  -> sparse typed-register write
```

The register graph remains non-Transformer:

```text
(a,b,op1) -> partial -> (partial,c,op2) -> final -> readout
```

## Deterministic full-domain all-pairs results

All scores below use the same `64^3` operands and all nine ordered operator
pairs. The 100M and 300M columns increase virtual circuit capacity while
reusing factor rows.

| Virtual scale | Virtual circuits | Physical trainable params | Active estimate | Full all-pairs |
|---|---:|---:|---:|---:|
| 20M factorized | 1,408 | 2.60M | 1.79M | **96.22%** |
| 100M factorized | 7,800 | 5.71M | 1.79M | **96.25%** |
| 300M factorized | 23,600 | 12.64M | 1.79M | **96.40%** |

The 500M factorized global control reached 94.72%. A depth-capped router reached
94.81%, and factor-address routing reached 95.55%; neither recovered the 300M
reference. Those ablations are recorded in `V0_38_FACTOR_ADDRESS_ROUTER.md`,
`V0_39_FACTOR_PAIR_BILINEAR_ROUTER.md`, and
`V0_40_DEPTH_CAPPED_ROUTING.md`.

For comparison, V0.36 with independent rows reached 94.95% at 20M and
94.13% at 100M. The factorized bank removes the previous negative capacity
trend in this controlled seed and protocol.

## Hidden-stage results

Each hidden checkpoint starts from its corresponding all-pairs checkpoint and
uses 5,000 adaptation steps on the six visible operator pairs. This is a
post-exposure adaptation test, not strict zero-shot generalization.

| Virtual scale | Hidden pair full grid | add → multiply | multiply → add |
|---|---:|---:|---:|
| 20M factorized | **98.66%** | 98.72% | 98.60% |
| 100M factorized | 97.15% | 97.52% | 96.78% |
| 300M factorized | 96.39% | 95.19% | 97.59% |

At 300M, re-evaluating all nine pairs after hidden adaptation gives 98.73%,
so the hidden score drop is concentrated in the two held-out compositions and
is not broad catastrophic forgetting.

## Commands

The exact training commands are the six matching `train_composition.py`
commands recorded by the run JSON files under `results/runs/`. Full-grid
evaluation uses `evaluate_composition.py --grid-size 64 --batch-size 4096`.
A larger batch materializes too many factorized matrices at once on a 12 GiB
GPU and is not memory-safe.

## Interpretation

The decisive result is not that 300M has dramatically higher quality. It is
that 20M → 100M → 300M stays flat-to-positive while the active estimate stays
near 1.79M. Stored virtual capacity grows, but per-example computation remains
sparse and reusable factor rows receive gradients from many addresses.

The remaining weakness is hidden-pair scaling: 20M is strongest, while 100M
and 300M need a better adaptation curriculum, second seeds, or explicit
composition-balanced sampling before claiming a general scaling law.
