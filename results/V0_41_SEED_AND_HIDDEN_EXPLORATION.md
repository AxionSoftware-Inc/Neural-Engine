# V0.41 second seed and hidden exploration audit

Status: **completed; seed variance is measured and parent-growth resolves the 500M regression**

## Second-seed all-pairs control

The V0.37 20M factorized-global configuration was retrained with seed 18. The
protocol, 10,000 steps, optimizer, batch size, and deterministic `64^3` full
grid were unchanged.

| Seed | Full all-pairs |
|---:|---:|
| 17 | 96.22% |
| 18 | 96.73% |
| Mean | **96.48%** |

The 0.51-point spread is small relative to the improvement over the old
58.10% baseline, so the V0.37 all-pairs result is not a one-seed accident.

## Hidden-stage route exploration ablation

Both runs start from the seed-18 all-pairs checkpoint and adapt for 5,000
steps on the seven visible composition pairs.

| Hidden adaptation | Hidden pair full grid | add → multiply | multiply → add |
|---|---:|---:|---:|
| 5% exploration | 96.38% | 94.97% | 97.78% |
| 0% exploration | **97.16%** | **96.68%** | 97.63% |

After the 0% exploration adaptation, all nine pairs together score 98.43%.
Visible-pair performance remains high, so the improvement is not caused by
trading away the visible tasks.

For reference, the seed-17 5% exploration hidden score was 98.66%. This shows
that hidden adaptation has materially higher seed variance than the all-pairs
control. The no-exploration recipe is therefore the current hidden-stage
candidate, not yet a universal scaling claim.

## Second-seed scale extension

The same all-pairs protocol was also run at 100M and 300M with seed 18.

| Virtual scale | Seed 17 | Seed 18 | Mean | Spread |
|---|---:|---:|---:|---:|
| 20M | 96.22% | 96.73% | **96.48%** | 0.51 pp |
| 100M | 96.25% | 93.80% | 95.03% | 2.45 pp |
| 300M | 96.40% | 95.39% | 95.89% | 1.01 pp |

The 20M result is stable, but the larger banks show meaningful optimization
variance. Therefore the factorized architecture is accepted as the best
capacity representation, while a single 100M/300M seed must not be presented
as a precise scaling law.

## Parent-growth follow-up

Because scratch 500M remained below 300M, a 300M seed-17 parent was expanded
into the 500M factorized geometry and trained for 10,000 more steps. The
deterministic full-domain score rose from 94.72% for scratch 500M to **99.66%**
for parent-growth, exceeding the 300M scratch control at 96.40%. A 5,000-step
hidden adaptation from that checkpoint reached **99.32%** on the two held-out
pairs and **99.77%** on all nine pairs after adaptation. Full details and
commands are in `results/V0_42_PARENT_GROWTH_FACTORIZED.md`.

## Decision

Keep 5% exploration for all-pairs training, and use 0% exploration for hidden
adaptation experiments. Parent-growth is now the preferred way to open larger
factorized capacity; scratch 500M is retained only as a negative control. A
second-seed parent-growth run and an out-of-distribution composition test are
required before treating 700M/1B as a quality scaling claim.
