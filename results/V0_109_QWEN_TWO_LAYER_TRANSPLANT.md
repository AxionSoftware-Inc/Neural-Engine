# V0.109 Qwen Two-Layer Parent Transplant

## Question

Does the positive single-FFN transplant signal survive when two adjacent late
Qwen FFNs are handed over to attention-free NE function blocks?

## Protocol

This is the next gate after V0.108. Qwen3-0.6B remains frozen except for two
new child blocks. First, layer 25's child is distilled from the original
layer-25 MLP. Layer 25 is then replaced by that child, and layer 26's child is
distilled from the layer-26 parent under the changed upstream representation.
Both children use

`LayerNorm(1024) -> Linear(1024,384) -> GELU -> Linear(384,1024)`

and no attention. Eight training batches and four held-out evaluation batches
are used. The final model is evaluated with the same alpha for both layers,
plus one-child controls.

```powershell
python -u benchmark_qwen_two_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --first-layer 25 --second-layer 26 --inner-size 384 --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_two_layer_transplant_seed2026.json
```

Hardware: NVIDIA GeForce RTX 3060 12GB, driver 591.86.

## Results

Teacher mean CE on the four held-out batches was 2.4080. Each child was much
smaller than its parent: 789,888 versus 9,437,184 scalar parameters, or
8.37% per layer.

| variant | parent contribution | CE delta | teacher top-1 agreement | decision |
|---|---|---:|---:|:---:|
| first child only | layer 25 = 0 | +0.0433 | 95.41% | pass |
| second child only | layer 26 = 0 | +0.0127 | 97.17% | pass |
| both, alpha 0.75 | 25/26 = 75% | +0.0027 | 98.54% | pass |
| both, alpha 0.50 | 25/26 = 50% | +0.0279 | 97.66% | pass |
| both, alpha 0.25 | 25/26 = 25% | +0.1808 | 94.82% | fail |
| both, alpha 0 | **full child handoff** | **+0.2377** | **91.50%** | **fail** |

The local hidden-state MSEs after multi-batch training were 1.891 for layer 25
and 2.684 for layer 26. The second child was trained after the first child was
installed, so this is not simply a stale-input measurement.

## Decision

**Naive independent full handoff is rejected for two adjacent layers.** The
single-layer V0.108 result does not compose automatically. The failure is
consistent with accumulated representation drift: the first child changes the
input distribution seen by the second child, and local MSE does not directly
optimize the final language logits. It is not yet evidence of a fundamental
architecture impossibility, because the 50% handoff remains within the local
quality gate and each one-child control passes.

The next experiment should keep the same two child blocks but perform joint
end-to-end logit distillation while the Qwen parent and attention layers stay
frozen. If that cannot recover alpha=0, the next repair should add an explicit
interface/state calibration layer; scaling the model or adding more cells
would be premature.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_transplant_seed2026.json`
