# V0.108 Qwen Parent-Function Transplant

## Question

Can one frozen Qwen3 FFN be replaced by a small attention-free Neural Engine
function block through a progressive parent-to-child handoff, without
retraining the rest of the Transformer?

This is the minimal gate for `taklif16`. It does not claim that the whole Qwen
model has been converted. It tests whether a local function can cross the
architecture boundary while the surrounding Transformer remains unchanged.

## Protocol

Qwen3-0.6B is loaded locally in float32 on CUDA. One late MLP (`layer_index`
25 or 26) is frozen as the parent. The child is an attention-free
`LayerNorm -> Linear(1024,384) -> GELU -> Linear(384,1024)` block. It is
trained for 300 steps on the parent's hidden-state input/output pairs. During
evaluation, the model sweeps

`output = alpha * parent(hidden) + (1 - alpha) * child(hidden)`

from the unchanged Transformer context, and checks the alpha=0 child-only
model against the original teacher logits.

Commands:

```powershell
python -u benchmark_qwen_parent_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-index 26 --inner-size 384 --batch-size 4 --sequence-length 64 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_parent_transplant_layer26.json
python -u benchmark_qwen_parent_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-index 26 --inner-size 384 --batch-size 4 --sequence-length 64 --steps 300 --learning-rate 0.003 --seed 2027 --output results/runs/qwen_parent_transplant_layer26_seed2027.json
python -u benchmark_qwen_parent_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-index 25 --inner-size 384 --batch-size 4 --sequence-length 64 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_parent_transplant_layer25.json
```

Hardware: NVIDIA GeForce RTX 3060 12GB, driver 591.86.

## Results

| layer | seed | teacher CE | child-only CE delta | child-only top-1 agreement | gate |
|---:|---:|---:|---:|---:|:---:|
| 26 | 2026 | 2.4635 | **+0.0103** | **98.44%** | pass |
| 26 | 2027 | 2.4635 | **+0.0394** | **96.09%** | pass |
| 25 | 2026 | 2.4635 | **+0.0183** | **94.53%** | pass |

The parent FFN has 9,437,184 scalar parameters. The child has 789,888,
8.37% of the parent's scalar count (about 11.95x fewer). The progressive
handoff remained finite and close to the parent across all three runs. The
layer-26 seed-2026 run reached its lowest CE at alpha=0.75, so this test also
shows that a mixed handoff can be useful before full replacement.

## Decision

**Positive local signal; keep and extend the direction.** The cross-architecture
function transplant is not rejected: a small attention-free block can reproduce
one Qwen FFN well enough on this local held-out text gate. This is materially
different from the earlier learned-basis attempt, which tried to copy Qwen
weights into a fixed NE basis and failed its teacher-logit gate.

The result is not yet evidence of end-to-end language quality, long-context
stability, or scalable replacement of many layers. The local eval text is
short and narrow, and the child was optimized for hidden-state regression.
The next gate should therefore use multiple text batches and progressively
replace two adjacent late FFNs, while keeping the parent, child, and alpha
ablations visible. Only after that passes should this be connected to a Neural
Engine circuit bank or tested at larger model sizes.

## Artifacts

- `benchmark_qwen_parent_transplant.py`
- `results/runs/qwen_parent_transplant_layer26.json`
- `results/runs/qwen_parent_transplant_layer26_seed2027.json`
- `results/runs/qwen_parent_transplant_layer25.json`
