# V0.42 parent-growth into 500M factorized capacity

Status: **positive capacity-conversion result; 500M is now useful when grown
from a trained 300M parent**

## Question

The V0.37 factorized bank made the active path nearly constant while increasing
virtual capacity, but the scratch 500M run regressed to 94.72% compared with
96.40% at 300M. This experiment tests whether the regression is optimization
and credit assignment rather than an intrinsic capacity limit:

1. train the 300M factorized model;
2. expand its factorized circuit/address rows into the 500M geometry;
3. copy the trained parent prefix into the larger model;
4. continue training under the same full all-pairs protocol;
5. compare against the existing scratch 500M checkpoint on the identical
   deterministic `64^3` grid.

This is still the typed-register, non-Transformer architecture. The model uses
explicit operand/partial/final/readout registers, multiplicative operand
interaction, a reusable factorized circuit bank, and eight active circuits per
stage.

## Warm-start construction

The new `grow_factorized_capacity.py` utility creates the target model from
the target YAML and copies equal tensors plus the leading rows of expanded
tensors. Thus the trained 300M controller, route keys, factor rows, and
factor-address codes are preserved; newly allocated rows retain the target
model initialization.

```powershell
python grow_factorized_capacity.py `
  --parent-checkpoint results/checkpoints/ne_typed_register_300m_factorized_all_10000.pt `
  --target-config configs/ne_typed_register_500m_factorized_all.yaml `
  --output results/checkpoints/ne_typed_register_500m_factorized_growth_init.pt
```

The generated target has 18,973,217 parameters. The parent prefix contains
23,600 route rows, 197 factor rows, and 23,600 factor-mix rows; the target has
38,600 route/mix rows and 197 factor rows. The growth report is stored inside
the ignored checkpoint for reproducibility.

## Training and deterministic audit

The warm-start model was trained for 10,000 steps on the RTX 3060 with the
same optimizer and batch protocol used by the scratch controls:

```powershell
python train.py `
  --config configs/ne_typed_register_500m_factorized_all.yaml `
  --init-checkpoint results/checkpoints/ne_typed_register_500m_factorized_growth_init.pt `
  --steps 10000 --device cuda `
  --run-id ne_typed_register_500m_factorized_growth_10000 `
  --output results/runs `
  --checkpoint results/checkpoints/ne_typed_register_500m_factorized_growth_10000.pt
```

The full-domain evaluator uses the same question set for every model:

```powershell
python evaluate_composition.py `
  --checkpoint results/checkpoints/ne_typed_register_500m_factorized_growth_10000.pt `
  --grid-size 64 --batch-size 4096 --device cuda `
  --pair add_then_add --pair add_then_subtract --pair add_then_multiply `
  --pair subtract_then_add --pair subtract_then_subtract `
  --pair subtract_then_multiply --pair multiply_then_add `
  --pair multiply_then_subtract --pair multiply_then_multiply
```

### Full all-pairs result

| Model | Parameters | Full `64^3` all-pairs | Active estimate |
|---|---:|---:|---:|
| 300M scratch, seed 17 | 12,638,321 | 96.40% | 1,792,305 |
| 500M scratch, seed 17 | 18,973,217 | 94.72% | 1,792,305 |
| **500M grown from 300M, seed 17** | **18,973,217** | **99.66%** | **1,792,305** |

The parent-growth run reaches **99.6569%** (`261,244 / 262,144` examples
correct on average across the nine pairs). It improves over scratch 500M by
4.94 percentage points and over the 300M scratch result by 3.26 points, while
keeping the estimated active path unchanged.

Per-pair full-grid accuracy:

| Pair | Accuracy |
|---|---:|
| add → add | 99.76% |
| add → subtract | 99.79% |
| add → multiply | 99.63% |
| subtract → add | 99.54% |
| subtract → subtract | 99.58% |
| subtract → multiply | 99.47% |
| multiply → add | 99.78% |
| multiply → subtract | 99.79% |
| multiply → multiply | 99.57% |

## Hidden-pair adaptation

Starting from the grown 500M all-pairs checkpoint, a 5,000-step adaptation was
run on the seven visible pairs with routing exploration disabled. The two
held-out pairs were then evaluated on the complete `64^3` grid:

| Checkpoint | Hidden pairs | All nine pairs after adaptation |
|---|---:|---:|
| 500M parent-growth, 5k hidden adaptation | **99.32%** | **99.77%** |

Hidden-pair details are 98.85% for add → multiply and 99.79% for multiply →
add. The all-nine post-adaptation score is 99.7696%; it does not indicate that
the held-out pairs were trained during adaptation—the model starts from an
all-pairs checkpoint, so this remains a post-exposure adaptation protocol just
like the earlier hidden-stage audits.

## Decision

**GO for the parent-growth recipe; NO-GO for scratch scaling as a quality
claim.** The negative scratch-500M result was not evidence that factorized
capacity itself is harmful. It was evidence that newly allocated capacity is
not automatically converted into useful circuits. Copying a trained parent
solves that optimization/credit-assignment discontinuity in this controlled
experiment.

The result is strong enough to make 500M the current engineering checkpoint,
but it is not yet a general scaling law. Before spending another overnight run
on 700M/1B, repeat parent-growth with a second seed and add an out-of-
distribution composition test. Keep the 300M scratch and 500M scratch results
as controls; do not replace them with the warm-start number.

## Reproducibility metadata

- Hardware: NVIDIA GeForce RTX 3060, CUDA.
- Parent: `results/checkpoints/ne_typed_register_300m_factorized_all_10000.pt`.
- Target: `configs/ne_typed_register_500m_factorized_all.yaml`.
- Growth initializer: `results/checkpoints/ne_typed_register_500m_factorized_growth_init.pt`.
- Full training report: `results/runs/ne_typed_register_500m_factorized_growth_10000.json`.
- Hidden training report: `results/runs/ne_typed_register_500m_factorized_growth_hidden_5000.json`.
- Checkpoints and JSON run artifacts are ignored by Git; commands and results
  are versioned in this report.
