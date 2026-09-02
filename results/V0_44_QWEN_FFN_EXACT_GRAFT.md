# V0.44 Qwen-style FFN exact circuit graft pilot

Status: **exact circuit conversion passes; sparse routing is not yet a
quality win**

## Question

Can a Qwen-style gated FFN be split into Neural Engine circuits so that all
circuits reproduce the source output exactly, then only a small subset is
executed for an input?

The canonical source for this experiment is `origin/main` commit `1fde5da`,
particularly `taklif9.md` and `taklif10.md`. The repository's current
`MicroCircuitBank` has one GELU projection, while Qwen uses two projections
and a SiLU product. Therefore the first implementation is an isolated gated
circuit bank; forcing the weights into the old GELU-only bank would not be an
exact transplant.

## Implementation

`neural_engine/pretrained_transfer.py` implements `SwiGLUCircuitBank`:

```text
source:  down_proj(silu(gate_proj(x)) * up_proj(x))
bank:    sum_i down_i(silu(gate_i(x)) * up_i(x))
```

The intermediate dimension is split into contiguous chunks. The last chunk
is zero-padded and masked when necessary. Optional Linear biases are copied;
the source `down_proj` bias is added once after the circuit sum.

`benchmark_qwen_transfer.py` is an optional real-checkpoint runner. It does
not make Transformers a repository dependency and reports full-model logit
equivalence when a local/cache Qwen checkpoint is available.

## Synthetic Qwen-shaped control

The local machine initially had only PyTorch. The optional `transformers` and
`safetensors` packages are now installed, but the Qwen weight blob is not
cached and the online download stalled at a 0-byte incomplete file. A small
synthetic module with the same Qwen SwiGLU projection algebra was therefore
used as the deterministic pretrained-weight-independent correctness control.

Commands:

```text
python -m pytest -q tests/test_pretrained_transfer.py
python benchmark_pretrained_transfer.py --seed 2026 --chunk-size 32 --router-steps 800
```

All existing tests and the new tests pass: **57 passed**. Three conversion
checks gave the following maximum absolute all-circuit error:

| Seed / chunk | Max absolute error | Mean absolute error |
|---|---:|---:|
| 2026 / 32 | 2.98e-7 | 2.15e-8 |
| 2027 / 32 | 3.87e-7 | 2.23e-8 |
| 2028 / 16 | 2.98e-7 | 2.19e-8 |

The copied bank has exactly 49,152 parameters for the 64→256 synthetic FFN,
matching the three source matrices. This confirms that the conversion is a
copy/re-index operation, not a lossy approximation.

The cached Qwen3-0.6B config was also used for a real-shape integration run:
28 layers, hidden size 1024, intermediate size 3072, and chunk size 256
(12 circuits per layer). With randomly initialized weights, the converted
model produced a maximum layer-MLP error of `2.38e-7` and full-logit error of
`2.26e-6` in float32; the full logits passed `allclose(atol=1e-5, rtol=1e-5)`.
This validates Qwen's actual module shapes, but it is not a pretrained-quality
result. In float16, chunk-wise reduction changes rounding across 28 layers
(max logit difference `0.00842`), so float32 is the strict conversion gate and
fp16 must use a dtype-appropriate tolerance/perplexity control.

## Sparse upper bound and router pilot

The offline oracle selects chunks by computing every chunk contribution first.
It is explicitly an upper bound, not a deployable router.

For seed 2026, chunk size 32, and 2 of 8 circuits active:

| Route | MSE to exact FFN | Estimated FFN MAC fraction |
|---|---:|---:|
| Offline contribution oracle | 0.00595 | 25% |
| Learned 2-layer pilot router | 0.00788 | 25% |
| Random 2-circuit control | 0.00855 | 25% |

The learned router is better than random in this small control, but it
recovers only 36.9% of the oracle-selected circuit IDs and remains noticeably
worse than the oracle. Seed 2027 gives the same direction (0.00796 learned vs
0.00871 random), so this is a weak routing signal, not a quality GO.

The pilot router scores every chunk address. That is acceptable for a
diagnostic but does not satisfy the project's scalable routing requirement.
The production path still needs hierarchical/activation-aware addressing;
otherwise the router can become the dense bottleneck.

## Decision

1. **GO for proposal 9's algebraic gate:** Qwen FFN neurons can be copied into
   contiguous gated circuits exactly, including a non-divisible final chunk.
2. **NO-GO for immediate 25% sparse deployment:** exact conversion alone does
   not establish that 2/8 chunks preserve the source quality, and the learned
   router is only a weak improvement over random selection.
3. **Do not merge this bank into the current typed-register model yet.** The
   current model's GELU-only circuit primitive cannot express Qwen's SiLU gate
   exactly. The gated primitive must first be tested against a real Qwen
   checkpoint, then connected through an activation/register adapter.
4. Proposals 11–14 remain conditional on the real checkpoint gate. Scaling to
   a 1B model before that gate passes would spend compute without resolving
   the representation mismatch.

The real pretrained checkpoint gate is still pending: the online Qwen weight
download stalled at a 0-byte incomplete cache file in this environment. No
claim about Qwen language quality or perplexity is made here.

## Next experiment

Install the optional checkpoint tooling with:

```text
python -m pip install -r requirements-transfer.txt
python benchmark_qwen_transfer.py --model Qwen/Qwen3-0.6B --chunk-size 256
```

The script also supports an architecture-only integration test when only the
Qwen config is cached:

```text
python benchmark_qwen_transfer.py --model Qwen/Qwen3-0.6B --random-from-config --local-files-only --device cuda --dtype float16
```

The acceptance gate is full-model logits allclose at float32 tolerance. If it
passes, the next experiment is proposal 10: train an activation-aware sparse
router at 75/50/25/16/8% and compare against random and magnitude controls.
If it fails, stop the Qwen graft and move to the proposal 11 hybrid/adapter
path rather than trying to repair each model size independently.
