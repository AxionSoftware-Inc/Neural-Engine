# V0.187 Qwen signed group-output sketch router

## Question

The existing `group-energy` feature sees only nonnegative per-group energy,
while the true subset error depends on signed output interactions. Can a small
signed sketch of each copied group's output improve the learned pairwise cost
router without changing the Qwen neuron-preserving groups or correction body?

## Method

Qwen3-0.6B layers 25--26 use the existing contiguous E=8/K=4 copied-neuron
bank and rank-64 cross-group correction. For each group, the router probe
computes the group SwiGLU coefficient and projects its output through eight
fixed cosine coordinates. The 64 signed features are fed to the existing
36-output pairwise cost head. The target is normalized pairwise regret; child
training is 300 soft steps plus 200 hard steps at learning rate `3e-4`, with
100 router-supervision steps.

The benchmark uses float32 CUDA, 16 calibration batches from
`data/qwen_calibration.txt`, 4 held-out batches from `data/qwen_eval.txt`,
batch size 8, sequence length 128, and the same `hard_route_scale=4` as the
validated pairwise controls. The documented prior pairwise hidden-input
control was `+0.04996` on seed 2026; the comparison below uses that control
for direction only.

## Results

| Seed | Router input | Learned CE delta | Exact-subset match layer 25 / 26 | Mean subset regret layer 25 / 26 | Timing / parent | Gate |
|---:|---|---:|---:|---:|---:|:---:|
| 2026 | signed group sketch, D=8 | +0.05899 | 53.27% / 72.29% | 0.1240 / 0.0995 | 1.183x | FAIL |
| 2027 | signed group sketch, D=8 | +0.05469 | 53.54% / 72.02% | 0.1230 / 0.1061 | 1.182x | FAIL |

The sketch router is slightly worse than the prior pairwise hidden-input
control and adds probe work. It still leaves a large learned/oracle gap; the
signed features did not make the 70-subset ranking generalize reliably.

## Decision

**REJECTED for adoption.** Do not increase sketch dimension or expand this
router to eight layers. The measured Qwen bottleneck is not fixed by adding a
small output-space probe to the existing pairwise head. The default model,
copied groups, and inference API are unchanged; this remains an opt-in
diagnostic implementation.

This closes another cheap router-feature hypothesis. The useful reference is
still the raw-neuron cross-group path: K=6/direct-hard passes across eight
layers, while K=4 has a good exact oracle but an unstable learned router. The
next work should target a genuinely different route generalization mechanism
or a fused execution path, not more hand-designed router feature coordinates.

## Reproduction

```powershell
python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 4 --calibration-rank 64 `
  --calibration-mode cross-group --dispatch-mode grouped `
  --partition-mode contiguous --route-source pairwise-cost-router `
  --router-input group-sketch --router-sketch-dim 8 `
  --router-target pairwise-regret --router-supervision-steps 100 `
  --hard-route-scale 4 --steps 300 --hard-train-steps 200 `
  --hard-learning-rate 3e-4 --train-batches 16 --eval-batches 4 `
  --batch-size 8 --sequence-length 128 `
  --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --seed 2026
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `tests/test_group_sketch_router.py`
- `results/runs/qwen_group_sketch_router_2layers_e8k4_seed2026.json`
- `results/runs/qwen_group_sketch_router_2layers_e8k4_seed2027.json`
