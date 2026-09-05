# V0.138 Qwen Single-Layer Sparse Circuit Bank

## Question

Can the stable single-layer Qwen FFN transplant be turned into a reusable
attention-free circuit bank while executing only one selected circuit per
token?

## Protocol

Qwen3-0.6B layer 26 is frozen as the parent. The child is a bank of four
independent `LayerNorm -> Linear(1024,384) -> GELU -> Linear(384,1024)`
circuits plus a small router. Training uses the soft mixture over all four
circuits; evaluation uses hard top-1 selected-token dispatch. The surrounding
Qwen Transformer remains frozen. The gate is measured on four held-out
sequence batches for two seeds.

```powershell
python -u benchmark_qwen_single_layer_bank.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-index 26 --inner-size 384 --num-experts 4 --active-experts 1 --routing-temperature 1.0 --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_single_layer_bank4x1_seed2026.json
python -u benchmark_qwen_single_layer_bank.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-index 26 --inner-size 384 --num-experts 4 --active-experts 1 --routing-temperature 1.0 --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2027 --output results/runs/qwen_single_layer_bank4x1_seed2027.json
```

## Results

| seed | teacher CE | alpha=0 child CE delta | teacher top-1 agreement | expert-body fraction | quality gate |
|---:|---:|---:|---:|---:|:---:|
| 2026 | 2.4080 | **+0.0237** | **96.39%** | **25%** | pass |
| 2027 | 2.4080 | **+0.0257** | **95.51%** | **25%** | pass |

The four-expert bank stores 3,291,268 scalar parameters, while one expert
body stores 789,888 parameters (8.37% of the 9,437,184-parameter parent FFN).
The router is always evaluated, but only one of four expert bodies is called
for each token.

## Interpretation

This is the first stable Qwen result combining a reusable circuit bank with a
real hard sparse execution path on a single layer. Both seeds remain inside
the `+0.05` CE gate, and the selected-token implementation avoids evaluating
the unselected expert bodies.

The bank still only replaces one local FFN while the rest of the Transformer
provides context. It does not prove multi-layer handoff or complete attention
removal; V0.137's two-layer quality gate remains rejected.

## Decision

**GO for single-layer sparse circuit-bank integration.** Keep the bank at the
0.6B checkpoint and next measure grouped CUDA latency and memory against the
parent/single-child controls. Do not increase model size or replace adjacent
layers until this bank has a measured runtime advantage and remains stable on
independent text.

## Artifacts

- `benchmark_qwen_single_layer_bank.py`
- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_single_layer_bank4x1_seed2026.json`
- `results/runs/qwen_single_layer_bank4x1_seed2027.json`
