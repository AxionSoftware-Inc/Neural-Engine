# V0.144 Qwen Large Token-Block Grouped Speedup

## Question

Does the normless grouped circuit bank become faster when a realistic larger
token block amortizes sorting, padding, and kernel-launch overhead?

## Protocol

The V0.143 normless four-expert top-1 grouped bank is repeated on Qwen3-0.6B
layer 26 with batch size 8 and sequence length 128 (1,024 tokens per batch).
The child is distilled for 300 steps on the same captured layer I/O. Parent
and bank are measured over 20 CUDA warmup and 50 synchronized serial
iterations, both for the isolated MLP and for the full Qwen forward.

## Results

| seed | parent FFN mean ms | normless grouped bank mean ms | bank/parent | end-to-end bank/parent | quality gate |
|---:|---:|---:|---:|---:|:---:|
| 2026 | 3.480 | 1.609 | 0.462x | 1.010x | pass |
| 2027 | 2.854 | 1.267 | 0.444x | 1.067x | pass |

The alpha=0 CE deltas are `+0.0132` and `+0.0140`, with teacher top-1
agreement `97.17%` and `96.80%`. The active expert-body fraction is 25% and
the child has 3,283,076 parameters, 34.79% of the parent FFN scalar count.

## Interpretation

This is the first repeatable active-compute speedup in the Qwen transfer
series: the isolated sparse FFN is about 2.16x--2.25x faster at 1,024 tokens.
The small end-to-end result is expected because only one of Qwen's 28 layers
is replaced; the unchanged layers dominate the total. The result confirms
that the earlier small-batch neutral/slower timing was an overhead regime, not
a fundamental inability of the sparse circuit to run faster.

## Decision

**Accept the large-token grouped path as a performance GO for the next
architecture gate.** Keep the Qwen transplant normless and grouped. The next
test should replace two adjacent layers with the same path and measure whether
the FFN-level gain compounds without the V0.112 routing-quality regression.
Do not jump to 700M/1B yet; first establish multi-layer quality and timing on
the cached 0.6B teacher.

## Artifacts

- `benchmark_qwen_single_layer_bank.py`
- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_single_layer_bank4x1_grouped_nonorm_b8s128_seed2026.json`
- `results/runs/qwen_single_layer_bank4x1_grouped_nonorm_b8s128_seed2027.json`
