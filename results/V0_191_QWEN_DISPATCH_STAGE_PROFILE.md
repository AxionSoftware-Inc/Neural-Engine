# V0.191 Qwen dispatch-stage profile

## Question

Which part of selected-group inference causes the sparse Qwen bank to lose its
theoretical compute reduction: routing, token grouping, GEMM, or scatter?

## Protocol

Qwen3-0.6B layer 26 was captured once at batch 8 × sequence 128 in float32
CUDA. The child used E=8/K=6 contiguous groups, no correction training, and
the child was explicitly set to `eval()` so only the hard selected path ran.
The profile was taken on the isolated MLP input, not the whole transformer,
with CUDA activity enabled.

## Results

| Isolated target | CUDA time | Relative to dense parent |
|---|---:|---:|
| dense parent MLP | `3.935 ms` | `1.00x` |
| grouped selected bank | `7.564 ms` | `1.92x` |
| packed selected bank | `10.228 ms` | `2.60x` |

For grouped dispatch, the three batched projections together took about
`2.457 ms`. The remaining time was spread across metadata and movement:
`sort`/`argsort`, token index construction, `index`, `index_copy_`,
`index_add_`, allocation and reshaping. The `sort` CUDA total was `0.929 ms`
and the main `index` total `0.798 ms`. Packed dispatch was worse because
`nonzero`/index gathering and six separate expert loops added about `2.676 ms`
of `index` work before the GEMMs.

This corrects an earlier profiling attempt that accidentally left the child in
training mode; that attempt executed all experts and is not used as evidence.

## Decision

The current grouped implementation is the best available PyTorch backend, but
it is not a deployment speedup. Removing only Python loops or changing FP16
precision cannot close the gap. The next valid backend must combine ragged
token grouping, expert GEMM tiling and output accumulation in one grouped
kernel (CUTLASS/cuBLAS grouped or a Triton-capable environment). A custom
one-block-per-pair kernel is already rejected by V0.189.

This profile gives no new quality or capacity evidence and does not justify a
700M/1B run. P-006 remains active.

## Reproduction

```powershell
python -u benchmark_qwen_dispatch_profile.py --dispatch-mode parent
python -u benchmark_qwen_dispatch_profile.py --dispatch-mode grouped
python -u benchmark_qwen_dispatch_profile.py --dispatch-mode packed
```

The profile helper is `benchmark_qwen_dispatch_profile.py`.
