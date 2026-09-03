# V0.70 Structured Numeric State Ablation

## Question

V0.69 localized the broad-value failure to the numeric state/interface rather
than raw virtual capacity. This experiment adds a separate learned recurrent
numeric scratch state to the typed-write model. The channel is initialized from
the numeric operand and updated by a shared neural transition conditioned on
the primitive operation; it is not a hard-coded arithmetic oracle. Its
projection is supplied to routing and to the per-step output readout.

## Configuration

- Base: `ne_dynamic_300m_composition_typed_write_adapter_broad_values`
- Operand values: 0--7; ordinary integer arithmetic; output classes: 512
- Target offset: 64; held-out pairs: `add -> multiply`, `multiply -> add`
- Operation routing adapter: rank 16; typed write adapter: rank 16
- Sparse bank: 23,600 virtual circuits, 154 factor rows, top-8 active
- Numeric scratch widths tested: 16 and 64
- Steps: 3,000 screen; batch size: 512
- Device: NVIDIA GeForce RTX 3060, CUDA
- Source state: `9344598`

## Results

| numeric width | seed | train | held-out | add -> multiply | multiply -> add | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 17 | 100.00% | 71.29% | 78.13% | 64.45% | 1,702,320 |
| 16 | 18 | 100.00% | 69.34% | 76.17% | 62.50% | 1,702,320 |
| 16 mean | -- | 100.00% | **70.31%** | 77.15% | 63.48% | -- |
| 64 | 17 | 100.00% | 70.31% | 74.90% | 65.72% | 1,751,952 |
| 64 | 18 | 100.00% | 70.21% | 75.98% | 64.45% | 1,751,952 |
| 64 mean | -- | 100.00% | **70.26%** | 75.44% | 65.09% | -- |

The 16D and 64D models contain 3,504,035 and 3,553,667 stored parameters,
respectively. Their active fractions are 48.58% and 49.30%. The broad-value
typed-write baseline is 69.17% mean at the same 3,000-step screen; the
hybrid-Fourier screen is 68.92%.

## Decision

The separate numeric channel gives a small positive signal at width 16, but
the gain is not large and does not survive increasing the channel to 64. It is
therefore **not accepted as the default architecture** and is not evidence for
scaling to 500M/700M/1B. The result confirms that merely adding or widening a
working-state channel does not fix broad-value composition.

The current strongest reference remains V0.67's typed-write model on the
narrow non-modular task and V0.68's long-depth validation. The next useful
architecture change must improve the operation/value composition interface
itself (for example, an explicit structured transition or circuit family),
not just add more state dimensions.

JSON evidence is stored in `results/runs/` under the
`typed_write_adapter_numeric_state_broad_values` and
`typed_write_adapter_numeric_state64_broad_values` run IDs.
