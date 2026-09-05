# V0.145 Qwen Two-Layer Joint Refinement

## Question

Can a short full-model joint refinement repair the composition error that
appears when two individually acceptable routed children replace adjacent
Qwen FFNs?

## Protocol

Layers 25 and 26 of Qwen3-0.6B are replaced by two independent four-expert
top-2, normless grouped circuit banks. Each child first receives the standard
300-step local FFN distillation. The pair is then refined jointly against the
frozen teacher logits for 50 full-model steps, using learning rate `1e-3` and
temperature `2.0`. The quality gate evaluates the combined
`shared_alpha_0` path on the same four 4×64 evaluation batches.

The earlier 8×128 joint run was stopped before producing a result after memory
pressure made it impractically slow: the script retains large teacher-logit
batches on the GPU. This 4×64 screening keeps the same architecture question
while fitting the RTX 3060 reliably.

## Results

| seed | combined alpha=0 CE delta | combined top-1 agreement | quality gate |
|---:|---:|---:|:---:|
| 2026 | +0.0433 | 93.07% | pass |
| 2027 | +0.0273 | 93.07% | pass |

Without joint refinement, the same normless grouped top-2 setup produced
`+0.0993` on seed 2026 at 8×128 and `+0.0436` on seed 2027. The joint step
therefore removes the clear seed-2026 composition failure in this screening;
the gain is not attributed to routing capacity or forced active paths.

The individual controls are also informative. On seed 2026 the first-child-only
control remains `+0.0546` after joint training while the second-child-only
control is `−0.0158`; on seed 2027 they are `+0.0372` and `−0.0152`. The
combined gate is the primary criterion, but this shows that joint training
does not make every local child independently perfect.

## Interpretation

The main two-layer issue is not simply that each child lacks capacity. The
local children can pass separately while their composed interface drifts;
short teacher-logit refinement can correct that drift. This is a credible
path toward multi-layer replacement, but the correction is an additional
training phase and must be measured for cost and stability.

## Decision

**Accept joint refinement as the current composition fix for the next gate.**
Keep the local 300-step distillation plus a short 50-step joint screen as the
default two-layer protocol. Before moving to 700M/1B or many layers, repeat
the pair with a timing harness and test whether joint refinement remains
stable at larger token blocks. Optimize the teacher-logit memory path before
attempting long joint runs.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_joint50_b4s64_seed2026.json`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_joint50_b4s64_seed2027.json`
