# V0.28 typed-register architecture and 20M/50M/100M/300M scale audit

V0.28 tests the architecture hypothesis from V0.27: the original model had
stored capacity, but no explicit working-memory semantics for composing two
primitive operations.  The new `TypedRegisterNeuralEngine` is a separate
non-Transformer control and uses the fixed execution graph:

```text
(a, b, op1) -> partial -> (partial, c, op2) -> final -> readout
```

The operands are encoded into typed slots, operation IDs and stage IDs are
added to every router query, and the selected circuit bank is executed
serially.  There is no attention and no Transformer block.  The model has
three explicit learned register writes: `partial`, `final`, and `readout`.

## Protocol

Every size uses the same composition benchmark, seed 17, numeric/Fourier
input, 8 active circuits, 3 execution stages, 5% training-only route
exploration, and stage supervision weight 0.5.  Each model first sees all
nine ordered operation pairs for 10,000 steps with batch size 128, then the
same checkpoint is fine-tuned for 5,000 steps on the six visible pairs.  The
two hidden pairs are `add_then_multiply` and `multiply_then_add`.

The primary quality metric is the deterministic `16^3 = 4,096` operand grid
per pair from `evaluate_composition.py`, not a small random evaluation batch.
All sizes therefore receive the same 1.28M all-pairs training samples and
640k hidden-stage samples.  The reported active estimate includes shared
controller/router work and only the selected circuit rows.

## Commands

```powershell
python train_composition.py --config configs/ne_typed_register_20m_all.yaml --steps 10000 --device cuda --run-id ne_typed_register_20m_all_10000 --output results/runs --examples-per-task 128 --log-every 1000 --checkpoint results/checkpoints/ne_typed_register_20m_all_10000.pt
python train_composition.py --config configs/ne_typed_register_20m.yaml --init-checkpoint results/checkpoints/ne_typed_register_20m_all_10000.pt --steps 5000 --device cuda --run-id ne_typed_register_20m_curriculum_hidden_5000 --output results/runs --examples-per-task 512 --log-every 500 --checkpoint results/checkpoints/ne_typed_register_20m_curriculum_hidden_5000.pt
python train_composition.py --config configs/ne_typed_register_50m_all.yaml --steps 10000 --device cuda --run-id ne_typed_register_50m_all_batch128_10000 --output results/runs --examples-per-task 128 --log-every 1000 --checkpoint results/checkpoints/ne_typed_register_50m_all_batch128_10000.pt
python train_composition.py --config configs/ne_typed_register_50m.yaml --init-checkpoint results/checkpoints/ne_typed_register_50m_all_batch128_10000.pt --steps 5000 --device cuda --run-id ne_typed_register_50m_curriculum_hidden_5000 --output results/runs --examples-per-task 512 --log-every 500 --checkpoint results/checkpoints/ne_typed_register_50m_curriculum_hidden_5000.pt
python train_composition.py --config configs/ne_typed_register_100m_all.yaml --steps 10000 --device cuda --run-id ne_typed_register_100m_all_10000 --output results/runs --examples-per-task 128 --log-every 1000 --checkpoint results/checkpoints/ne_typed_register_100m_all_10000.pt
python train_composition.py --config configs/ne_typed_register_100m.yaml --init-checkpoint results/checkpoints/ne_typed_register_100m_all_10000.pt --steps 5000 --device cuda --run-id ne_typed_register_100m_curriculum_hidden_5000 --output results/runs --examples-per-task 512 --log-every 500 --checkpoint results/checkpoints/ne_typed_register_100m_curriculum_hidden_5000.pt
python train_composition.py --config configs/ne_typed_register_300m_all.yaml --steps 10000 --device cuda --run-id ne_typed_register_300m_all_10000 --output results/runs --examples-per-task 128 --log-every 1000 --checkpoint results/checkpoints/ne_typed_register_300m_all_10000.pt
python train_composition.py --config configs/ne_typed_register_300m.yaml --init-checkpoint results/checkpoints/ne_typed_register_300m_all_10000.pt --steps 5000 --device cuda --run-id ne_typed_register_300m_curriculum_hidden_5000 --output results/runs --examples-per-task 512 --log-every 500 --checkpoint results/checkpoints/ne_typed_register_300m_curriculum_hidden_5000.pt
```

The deterministic grid commands were:

```powershell
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_20m_all_10000.pt --grid-size 16 --batch-size 64 --device cuda
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_20m_curriculum_hidden_5000.pt --grid-size 16 --batch-size 64 --device cuda
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_50m_all_batch128_10000.pt --grid-size 16 --batch-size 64 --device cuda
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_50m_curriculum_hidden_5000.pt --grid-size 16 --batch-size 64 --device cuda
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_100m_all_10000.pt --grid-size 16 --batch-size 64 --device cuda
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_100m_curriculum_hidden_5000.pt --grid-size 16 --batch-size 64 --device cuda
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_300m_all_10000.pt --grid-size 16 --batch-size 64 --device cuda
python evaluate_composition.py --checkpoint results/checkpoints/ne_typed_register_300m_curriculum_hidden_5000.pt --grid-size 16 --batch-size 64 --device cuda
```

## Results

| Model | Stored params | Active estimate | All-pairs grid | Hidden-pair grid | All-pairs time | Hidden time |
|---|---:|---:|---:|---:|---:|---:|
| Typed Register 20M | 19.78M | 1.510M (7.64%) | 57.67% | **65.56%** | 10.0 min | 4.9 min |
| Typed Register 50M | 51.01M | 1.513M (2.97%) | 56.48% | 63.34% | 16.8 min | 8.4 min |
| Typed Register 100M | 103.24M | 1.513M (1.47%) | **57.93%** | **89.87%** | 27.0 min | 13.5 min |
| Typed Register 300M | 309.52M | 1.517M (0.49%) | 57.38% | 89.20% | 67.6 min | 33.7 min |

The 20M all-pairs control is 57.67%, versus 31.18% for the earlier
non-register NE control at 20,000 steps.  On hidden pairs, the old audited
300M curriculum mean was 13.79% ± 0.96; the new register model reaches
65.56% at 20M, 63.34% at 50M, 89.87% at 100M, and 89.20% at 300M under the
single-seed protocol above.  The new architecture is therefore a large
positive result, while the 100M checkpoint is the current quality winner.

## Interpretation

The experiment supports the architectural diagnosis: explicit dataflow and
typed operator routing matter more than simply appending circuit rows.  The
stored bank grows from 19.78M to 309.52M while the active estimate remains
near 1.51M, so sparse activation is preserved.  The sharp jump appears at
100M after the same exposure, but 300M does not improve over 100M; more bank
capacity still does not guarantee monotonic quality.

The immediate next experiment should not be 700M/1B.  First validate the
100M result with at least one additional seed and inspect whether route
families are reused by primitive operator rather than memorized by operand
tuple.  Then test route sharing or a compact operator-specific bank.  The
300M run is useful as a feasibility point, but its 101-minute local training
cost and slightly lower hidden grid make it inferior to 100M for now.

All measurements were run on an NVIDIA GeForce RTX 3060 with 12 GiB VRAM.
The current implementation and configs are on the
`exp/typed-register-composition` branch.  The pre-change audited state is
frozen on `freeze/v0.27-audited`.
