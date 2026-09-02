# V0.47 learned circuit basis and layer-adaptive routing audit

Status: **learned redundant basis is rejected at the tested budgets; static
per-layer sensitivity screening is not reproducible; no deployable Qwen
sparsity architecture is accepted yet**.

## Question

V0.46 established two constraints. First, contiguous Qwen FFN chunks are not a
redundant representation: a local contribution oracle still needed nearly all
chunks for teacher-level fidelity. Second, a compact nonlinear FFN helped only
in a small late-layer pilot. V0.47 tests two ways to address the suspected
architectural problem:

1. learn a redundant latent circuit basis by exposing trainable circuit banks
   to random active masks; and
2. keep sensitive layers dense while sparsifying only individually low-impact
   layers.

These are fidelity controls on the cached real `Qwen/Qwen3-0.6B` checkpoint,
not a general language benchmark. The teacher CE is comparable only within a
single run. The local contribution oracle computes every circuit before
selecting a route, so it is an upper bound and not a runtime speed result.

## Learned redundant circuit basis

`benchmark_qwen_learned_basis.py` converts selected Qwen FFNs into trainable
`SwiGLUCircuitBank` modules. During local distillation, a random fixed-size
mask is applied to the bank and the selected outputs are scaled by `N/k`.
Evaluation reports both the same random route and a local contribution oracle.
If the random route works, the representation has become redundant; if only
the oracle works, routing is still the bottleneck.

| Target layers | Chunk | Active | Steps | Teacher CE | Random CE delta | Oracle CE delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| late 6 | 64 | 25% | 300 | 5.0758 | +0.5212 | +0.7548 | **NO-GO** |
| late 6 | 64 | 50% | 50 | 6.5781 | +0.5039 | +0.0816 | **NO-GO** |

The 25% run directly rejects the intended claim: training a redundant basis
did not make an arbitrary quarter of circuits usable. The 50% run is a quick
screen rather than a final quality estimate, but it gives the same important
signal—random execution remains far from the teacher even when the oracle is
much closer. The per-layer training losses also diverged in the short control,
with the deepest tested layer remaining difficult to reconstruct. Increasing
steps or model size is therefore not justified before changing the objective
or representation again.

Reproduction commands:

```text
python benchmark_qwen_learned_basis.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --chunk-size 64 --active-fraction 0.25 --layer-indices 14,18,22,24,26,27 --steps 300 --learning-rate 0.001 --full-reconstruction-weight 0.05
python benchmark_qwen_learned_basis.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --chunk-size 64 --active-fraction 0.50 --layer-indices 14,18,22,24,26,27 --steps 50 --learning-rate 0.001 --full-reconstruction-weight 0.05
```

## Layer sensitivity and static schedules

`benchmark_qwen_layer_sensitivity.py` sparsifies one layer at a time with the
same local contribution oracle, ranks the 25% rows, and then applies the
ranked layers together. This tests whether a cheap fixed schedule can avoid
the error accumulation of uniform full-stack sparsity.

The first evaluation variant looked promising at the oracle level:

| Sparse layers | Effective target FFN active fraction | CE delta | Selected layers |
|---:|---:|---:|---|
| 4 | 89.29% | -0.1685 | 9, 10, 11, 15 |
| 8 | 78.57% | -0.1592 | 3, 9, 10, 11, 12, 13, 15, 22 |
| 12 | 67.86% | -0.0630 | 3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 21, 22 |
| 16 | 57.14% | -0.0212 | 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 21, 22, 24, 26 |

The independent second evaluation variant invalidated that fixed ranking:

| Sparse layers | Effective target FFN active fraction | CE delta | Decision |
|---:|---:|---:|---|
| 4 | 89.29% | -0.3105 | fragile oracle signal |
| 8 | 78.57% | -0.2721 | fragile oracle signal |
| 12 | 67.86% | +0.0422 | **fails quality gate** |
| 16 | 57.14% | +0.2542 | **NO-GO** |

The selected layer sets changed materially between the two short text
variants. Therefore, single-layer CE sensitivity is real but not a stable
proxy for a deployable global route. This proposal is rejected as a static
schedule. It also provides no speed claim because the oracle still evaluates
all circuit chunks.

Reproduction commands:

```text
python benchmark_qwen_layer_sensitivity.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --chunk-size 64 --active-fractions 0.25 0.50 --combination-counts 4 8 12 16 --eval-variant 0
python benchmark_qwen_layer_sensitivity.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --chunk-size 64 --active-fractions 0.25 --combination-counts 4 8 12 16 --eval-variant 1
```

## Decision log

- **GO:** exact Qwen FFN-to-circuit algebra. It remains useful as a transfer
  and analysis primitive.
- **NO-GO:** uniform 25% or 50% contiguous-chunk routing across the full Qwen
  stack.
- **NO-GO:** the current learned redundant basis at 25%; the short 50% screen
  also does not pass.
- **NO-GO:** a fixed layer schedule selected from single-layer sensitivity.
- **CONDITIONAL:** late-layer compact or low-rank blocks, but only after a
  global end-to-end training objective and independent multi-seed validation.

Do not move to a 700M, 1B, or 1.7B Qwen transfer experiment based on these
results. More parameters cannot repair a representation/router failure that
already appears in the 0.6B teacher. The independent Neural Engine
factorized-capacity line remains the stronger accepted result.

## Next experiment

The next credible test is a global layer-gate student, not another per-layer
ranking heuristic:

1. preserve early sensitive layers and initialize a compact late-layer bank;
2. train the layer gate end-to-end against teacher logits plus a controlled
   active-compute penalty;
3. evaluate on larger independent text, two evaluation variants, and at least
   three seeds; and
4. measure a grouped/structured kernel separately from fidelity.

The gate must be allowed to choose the natural active budget for each token;
it must not be forced to use a fixed number of active parameters. A pass
requires teacher top-1/KL fidelity, stable behavior across variants and
seeds, and actual lower latency. If this fails, Qwen neuron transfer should
be kept for initialization/analysis only and further scale experiments should
return to the independent Neural Engine architecture.
