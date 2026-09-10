# P-003 native compiled-execution probe

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Device: NVIDIA GeForce RTX 3060, CUDA 12.4

## Question

Can the current native fixed-width and learned-width paths use the local
`torch.compile`/Inductor backend to remove Python and dispatch overhead?

## Protocol

- PyTorch `2.6.0+cu124`, `torch.compile(mode="reduce-overhead")`.
- One 500M stable-prefix seed17 checkpoint.
- Fixed K=8, fixed K=16, and learned grouped K=8/K=16 controls.
- Balanced batch of 480 examples, three warm-ups and ten timed repeats.
- Eager timings were collected before compilation; a backend error in one
  variant was caught so the remaining controls could still be inspected.

## Result

All three variants reached the eager timing stage, but Inductor compilation
failed with the same environment error:

`BackendCompilerFailed: Cannot find a working triton installation.`

This is a toolchain blocker, not a model-parity or quality failure. No compiled
latency claim is made and no compiler-dependent default was changed.

## Decision

`BLOCKED LOCALLY — DO NOT INSTALL OR CHANGE DEFAULT TOOLCHAIN AUTOMATICALLY`.

The next runtime implementation must be a static-index or custom fused path
that does not depend on the unavailable Triton installation. If a working
compiler environment is later provided, this probe can be rerun unchanged.

## Raw evidence and reproduction

- [Compile probe JSON](diagnostic_native_compile_s17_20260910.json)
- [Compile benchmark](../benchmark_native_compile.py)

```powershell
python benchmark_native_compile.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
  --include-learned --examples-per-task 32 --warmup 3 --repeats 10 `
  --output results/diagnostic_native_compile_s17_20260910.json
```
