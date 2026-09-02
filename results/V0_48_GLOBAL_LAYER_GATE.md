# V0.48 global token-adaptive layer gate audit

Status: **conditional research signal; not a 4x sparse-model pass**.

## Question

V0.47 rejected a fixed layer schedule because individually low-impact layers
changed between evaluation variants. V0.48 replaces that heuristic with a
token-level gate trained end-to-end against the frozen Qwen3-0.6B teacher. The
gate sees each selected layer's input and chooses whether to keep or skip the
complete FFN for that token. The active budget is therefore learned rather
than forced to a fixed count.

`benchmark_qwen_global_layer_gate.py` has two execution modes. The soft mode
is differentiable and is used for training. The hard mode thresholds the gate
and reports actual token keep fractions. The current wrapper still evaluates
the FFN before applying the hard mask, so these are quality/route results;
they are not latency results. A deployable implementation needs grouped
conditional kernels.

All numbers below are from the cached real `Qwen/Qwen3-0.6B` checkpoint. The
short text controls are not a general language benchmark. CE deltas can be
negative when skipping a teacher transformation improves the particular text,
so teacher top-1 agreement and logit MSE are also required.

## Late-six-layer Pareto screen

The selected layers are `14,18,22,24,26,27`. Overall FFN active fraction
counts the untouched 22 layers as dense; it is not the fraction of all model
parameters or total latency.

| Setup | Eval | Hard keep in selected 6 | Overall FFN active | CE delta | Teacher top-1 | Logit MSE | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| penalty 0.02, 80 steps, 8x16 | variant 1 | 94.53% | 98.83% | -0.0325 | 96.875% | 0.0467 | fidelity signal, tiny saving |
| penalty 0.05, 80 steps, 8x16 | variant 1 | 83.59% | 96.48% | -0.0647 | 86.719% | 0.2229 | quality gate fails |
| penalty 0.02, 60 steps, 4x8 | variant 1, seed 17 | 84.38% | 96.66% | -0.1143 | 90.625% | 0.2579 | short-control signal |
| penalty 0.02, 60 steps, 4x8 | variant 0, seed 17 | 84.38% | 96.66% | -0.0633 | 78.125% | 0.2255 | eval-sensitive |

The longer variant-1 control is the strongest result so far: 96.875% teacher
top-1 at only 1.17% FFN reduction. The higher penalty exposes the expected
trade-off, but the 3.52% FFN reduction point is already below the teacher
fidelity gate. The gate consistently learns to keep layer 27 and to drop more
of layer 24, which is a useful structural signal, but that does not establish
generalization.

The short variant-1 result is similar across seeds: seed 2026 gives 85.42%
selected-layer keep, CE delta `-0.1170`, and 90.625% top-1; seed 17 gives
84.38%, `-0.1143`, and 90.625%. Variant 0 is less stable, including the
78.125% seed-17 top-1 result. Larger independent data is still required.

## All-layer screen

Applying the gate to all 28 layers does not produce a useful sparse model:

| Eval variant | Hard overall FFN active | CE delta | Teacher top-1 | Logit MSE | Decision |
|---:|---:|---:|---:|---:|---|
| 0, 30 steps, 4x8 | 91.52% | -0.1098 | 62.5% | 1.1217 | **NO-GO** |
| 1, 30 steps, 4x8 | 93.64% | -0.4881 | 59.375% | 0.7669 | **NO-GO** |

This is only 6.36–8.48% FFN reduction, far from the proposed 4x active
budget, and teacher agreement is poor. The negative CE is not sufficient to
override that result because it is measured against the input labels, not
exact teacher-logit fidelity.

## Decision

- **GO as a research direction:** a global token-dependent layer gate can
  identify a small amount of removable late-layer work without a fixed active
  path.
- **NO-GO as a production architecture:** current savings are too small and
  the all-layer route fails teacher fidelity.
- **NO-GO for scale-up:** do not move this Qwen transfer gate to 700M, 1B, or
  1.7B yet. More parameters would not fix the missing conditional-kernel and
  generalization evidence.

The accepted Qwen result remains exact FFN algebraic conversion. The
independent Neural Engine factorized-capacity line remains the stronger
quality-scaling result.

## Next step

Run a real gate study with a larger independent corpus, at least three seeds,
both evaluation variants, and a structured execution prototype. Keep early
layers dense, train only a small gate/decoder, and select checkpoints by
teacher top-1/KL plus active FFN fraction—not CE alone. If the best stable
point cannot exceed roughly 10% real FFN savings without teacher-fidelity loss,
stop pursuing Qwen transfer as the main sparse architecture and return effort
to the independent Neural Engine circuit design.

Reproduction entry point:

```text
python benchmark_qwen_global_layer_gate.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-indices 14,18,22,24,26,27 --sequence-length 16 --train-batches 4 --eval-batches 8 --steps 80 --compute-penalty 0.02 --learning-rate 0.01 --eval-variant 1
```
