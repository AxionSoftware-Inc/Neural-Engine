# V0.46 teacher-distilled routing and structural compression audit

Status: **full-stack 4x sparse FFN is rejected; raw Qwen chunk sparsity is
representation-limited; late-layer compact FFN remains a small conditional
signal**.

## Question

V0.45 showed that exact Qwen3-0.6B FFN conversion works, but local chunk
routing does not preserve the teacher at 25% active FFN compute. V0.46 tests
the proposed next step: train routes against the global teacher logits, add a
small low-rank residual for omitted computation, and allow the active budget
to vary by token. A separate pilot tests a compact nonlinear latent FFN rather
than dropping contiguous Qwen neuron chunks.

All runs used the cached real `Qwen/Qwen3-0.6B` checkpoint on an RTX 3060
12-GB GPU. The text sets are small fidelity controls, not a general language
benchmark. The teacher CE therefore only has meaning within each run.

## Teacher-distilled residual sparse FFN

The new `TeacherDistilledSparseSwiGLU` module starts with an exact soft bank,
trains router/residual parameters against teacher top-k logits, then switches
to straight-through hard top-k. The Qwen bank and base stack remain frozen.
The residual down projection is zero-initialized. The training path is dense
by design; it is a research distillation path, not deployable routing.

| Target layers | Chunk | Active | Residual rank | Hard CE delta | Exact/hard latency | Decision |
|---|---:|---:|---:|---:|---:|---|
| late 6 | 256 | 25% | 64 | +0.578 | 34.6 / 44.6 ms | NO-GO |
| late 6 | 256 | 50% | 64 | +0.128 | 30.5 / 59.5 ms | quality close, speed NO-GO |
| late 6 | 64 | 25% | 64 | +0.051 | 26.2 / 44.3 ms | near-gate, not pass |
| late 6 | 64 | 25% | 256 | -0.428* | 25.9 / 50.2 ms | not reliable |
| all 28 | 64 | 25% | 64 | +4.902 | 25.0 / 62.4 ms | **NO-GO** |
| all 28 | 64 | 50% | 64 | +0.433 | 28.6 / 102.8 ms | quality/speed NO-GO |

The residual clearly helps: for late 6, chunk64 25% moves from `+0.459`
without residual to `+0.051` with rank64, and chunk256 50% moves from `+0.450`
to `+0.128`. It does not solve the full-stack problem. The rank256 `-0.428`
result has only 64 evaluation sequences and just 42% teacher top-1 agreement,
so it is treated as small-set label overfit, not a teacher-fidelity win.

The hard route is slower in every measurement because the current generic
per-token `einsum` implementation pays dynamic gather/dispatch overhead. A
deployable speed claim requires a grouped/structured kernel audit.

Reproduction entry point:

```text
python benchmark_qwen_distilled_residual.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --chunk-size 64 --active-fraction 0.25 --residual-rank 64 --layer-indices 14,18,22,24,26,27 --steps 60
```

## Adaptive active-budget oracle

`benchmark_qwen_adaptive_sparse.py` chooses the smallest per-token circuit
prefix whose local FFN output error is below a tolerance. It computes every
circuit first, so it is an upper bound and not a deployable router.

| Local relative error tolerance | Mean active fraction | Full-model CE delta |
|---:|---:|---:|
| 0.02 | 99.18% | +0.0065 |
| 0.05 | 96.85% | +0.0176 |
| 0.10 | 90.56% | +0.1347 |
| 0.20 | 74.96% | +0.3032 |
| 0.30 | 58.53% | +0.3360 |
| 0.50 | 32.55% | +0.8468 |

This is the strongest evidence that the current contiguous Qwen chunk
representation is the bottleneck: even an oracle needs almost all chunks for
teacher-level fidelity. A cheaper router cannot beat this upper bound.

## Compact nonlinear latent FFN

`benchmark_qwen_compact_ffn.py` replaces selected Qwen FFNs with a smaller
Qwen-shaped SwiGLU trained on teacher MLP input/output pairs.

| Target layers | Latent/original intermediate | Full-model CE delta | Teacher top-1 |
|---|---:|---:|---:|
| all 28 | 768/3072 (25%) | +6.756 | 4.7% |
| late 6 | 768/3072 (25%) | +0.119 | 70.3% |
| late 6 | 1024/3072 (33%) | +0.086 | 68.8% |

This is better than full-stack chunk dropping but still misses the `+0.02`
gate. Local MLP MSE improvement does not guarantee global teacher fidelity;
errors accumulate through the stack.

## Decision and next step

The V0.46 stop decision is:

- do not scale the current sparse-transfer recipe to 1B/1.7B;
- do not claim Transformer replacement or 4x Qwen quality retention;
- keep exact FFN graft as a valid algebraic transfer primitive;
- keep late-layer compact/low-rank replacement as a **conditional research
  direction**, not a result;
- preserve early layers and train a global end-to-end student on a larger,
  independent text/control set before changing the architecture again;
- require both teacher top-1/KL fidelity and an actually faster grouped kernel.

The next credible architecture experiment is a learned latent circuit basis:
train a small set of reusable basis circuits and a global output-aware decoder,
instead of assigning semantic meaning to contiguous slices of Qwen's FFN.
If that cannot pass the same gates at late layers, Qwen neuron transfer should
be considered useful for initialization/analysis only, while the independent
Neural Engine arithmetic architecture remains the main research track.
