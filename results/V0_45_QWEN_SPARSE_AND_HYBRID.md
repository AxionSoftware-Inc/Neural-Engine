# V0.45 Qwen pretrained sparse FFN and minimal hybrid audit

Status: **exact graft is valid; 4x sparse FFN/hybrid fails, while a selective
attention-transplant pilot is a limited positive**

## Scope

V0.44 proved that Qwen3-0.6B's SwiGLU FFN can be copied into contiguous gated
circuits exactly. This run tests the next canonical steps from
`origin/main` commit `1fde5da`:

- `taklif10.md`: learned sparse FFN routing;
- `taklif11.md`: original Qwen attention retained with sparse FFN circuits.

The teacher is the real cached Qwen3-0.6B checkpoint. All attention,
embeddings, RMSNorms, and the LM head stay unchanged. The route oracle first
computes every chunk contribution, then retains the largest chunks; therefore
it is an upper bound on a local contribution router, not a deployable runtime
measurement. Random selection is the matched active-count control.

## Real text control

The tokenizer encodes a short held-out English evaluation paragraph. The
metric is teacher-student next-token cross-entropy, plus logit MSE and top-1
agreement. It is a small fidelity control, not a general language benchmark.
The full teacher CE is **5.8151**.

### Chunk size 256: 12 circuits per layer

| Route | Active FFN fraction | CE delta | Logit MSE | Top-1 agreement |
|---|---:|---:|---:|---:|
| Full exact bank | 100% | +0.0000 | 0.0000 | 100.0% |
| Local contribution oracle | 75% | +0.0788 | 2.134 | 52.2% |
| Local contribution oracle | 50% | +1.0152 | 5.675 | 21.7% |
| Local contribution oracle | 25% | **+2.2548** | 9.033 | 8.7% |
| Random control | 25% | +8.2437 | 30.992 | 0.0% |

### Chunk size 64: 48 circuits per layer

| Route | Active FFN fraction | CE delta | Logit MSE | Top-1 agreement |
|---|---:|---:|---:|---:|
| Local contribution oracle | 75% | -0.0204 | 1.321 | 71.7% |
| Local contribution oracle | 50% | +0.4434 | 4.011 | 28.3% |
| Local contribution oracle | 25% | **+1.7035** | 7.883 | 13.0% |
| Random control | 25% | +12.8066 | 71.203 | 0.0% |

### Chunk size 32: 96 circuits per layer

| Route | Active FFN fraction | CE delta | Logit MSE | Top-1 agreement |
|---|---:|---:|---:|---:|
| Local contribution oracle | 75% | -0.0637 | 0.960 | 69.6% |
| Local contribution oracle | 50% | +0.0582 | 3.489 | 45.7% |
| Local contribution oracle | 25% | **+1.4704** | 6.985 | 13.0% |
| Random control | 25% | +10.5062 | 27.525 | 0.0% |

Chunking is therefore a real lever: smaller chunks make the 50% point much
better. But even the best local oracle at 25% active remains far from the
teacher, so finer routing alone does not reach the required 4x FFN reduction.

## Learned router pilot

For the canonical learned-routing check, a small two-layer router was trained
for each of the 28 Qwen layers. It saw frozen teacher hidden states and was
trained to predict the top-contribution chunk labels. It was then evaluated on
held-out text while all MLPs used hard top-3-of-12 routing (25% FFN).

| Route | CE | CE delta | Logit MSE | Top-1 agreement |
|---|---:|---:|---:|---:|
| Full teacher | 2.4080 | 0 | 0 | 100.0% |
| Local oracle | 4.2676 | +1.8596 | 4.876 | 51.6% |
| **Learned router** | **6.3357** | **+3.9277** | 6.987 | 18.8% |
| Random control | 13.1289 | +10.7209 | 56.062 | 1.2% |

The learned router is materially better than random, but worse than the local
oracle and does not preserve pretrained quality. The router also scores all
addresses, so this pilot has not yet demonstrated scalable hierarchical
routing. Its positive result is only that hidden-state routing contains
useful signal.

The different full-CE values in the two tables come from different sequence
lengths and calibration text used by the two commands; route deltas are only
compared within their own run.

## Proposal 12 — progressive attention transplant

Because the minimal hybrid gate failed at 25% active FFN, this was run only as
the canonical low-cost Level-1 local-block pilot. A middle or late Qwen
attention block was distilled into an attention-free GRU sequence mixer with
256 memory dimensions. The teacher's attention, all other layers, and the LM
head stayed frozen. Training used 300 steps on four 64-token sequences; the
held-out evaluation used a separate four-sequence text stream.

| Replaced attention layer(s) | Local student MSE | Full CE delta | Top-1 agreement |
|---|---:|---:|---:|
| Layer 0, recurrent mixer | 0.0185 | **+4.8367** | 30.5% |
| Layer 14, recurrent mixer | 0.0838 | -0.0019 | 96.1% |
| Layer 27, recurrent mixer | 1.9093 | -0.0153 | 99.2% |
| Layers 14 + 27, recurrent mixers | 0.0838 / 1.8112 | **-0.0234** | 96.1% |

The zero-attention controls for layers 0, 14, 27, and 14+27 had CE deltas of
`+10.9493`, `+0.0238`, `+0.0135`, and `+0.0351`, respectively. The learned
recurrent replacements therefore beat their zero controls in this small
distillation test. However, layer 0 is clearly sensitive, and the small text
set is not a general language benchmark. This is a **limited GO to continue
middle/late progressive stages**, not a GO to remove attention broadly.

## Decision by canonical proposal

### Proposal 9 — exact FFN decomposition: GO

Real Qwen3-0.6B full float32 conversion gives zero observed MLP and logit
error after the source Linear order is preserved. The reusable implementation
is in `neural_engine/pretrained_transfer.py`.

### Proposal 10 — sparse FFN routing: NO-GO at 4x, weak research signal

Both the contribution oracle and learned router beat random, but 25% active
does not retain teacher behavior. A learned router should not be scaled to
700M/1B yet. The next useful research target is teacher-distilled routing or
smaller structured chunks, not more capacity.

### Proposal 11 — minimal Qwen + NE hybrid: CONDITIONAL NO-GO

The measured model already is the minimal hybrid shape: original attention
plus routed gated FFN circuits. Retaining attention does not rescue 25% FFN
sparsity. The hybrid can still be a future endpoint after a retrained sparse
FFN passes the proposal10 gate; this run does not justify attention-frequency
reduction.

### Proposal 12 — progressive attention transplant: LIMITED GO

The one- and two-block pilots show that middle/late attention functions can be
approximated by a recurrent memory mixer on this small control, while the
first layer cannot. Continue only with middle/late sensitivity-ordered stages;
do not use all-layer or every-2nd-layer replacement yet.

### Proposal 13 — full Qwen → NE roadmap: NOT READY TO SCALE

The 0.6B checkpoint now passes exact conversion and provides a limited
attention-transplant signal, but useful 25% sparse FFN routing has not passed.
The proposal13 1.7B/4B/8B ladder is therefore not justified yet; scaling would
repeat the unresolved sparse-quality problem at higher cost.

### Proposal 14 — execution order and stop gates: FOLLOWED

The sequence was followed: exact conversion → sparse controls → minimal
hybrid → one/two-block attention pilot. The 4x sparse stop gate remains active;
deeper attention-frequency reduction and size scaling are not started. This
preserves negative evidence instead of producing architecture-only numbers
that could be mistaken for language quality.

## Reproduction

```text
python benchmark_qwen_transfer.py --model <local-qwen-snapshot> --local-files-only --device cuda --dtype float32 --chunk-size 256
python benchmark_qwen_sparse.py --model <local-qwen-snapshot> --tokenizer <local-qwen-snapshot> --local-files-only --device cuda --dtype float32 --chunk-size 256 --sequence-length 128
python benchmark_qwen_sparse.py --model <local-qwen-snapshot> --tokenizer <local-qwen-snapshot> --local-files-only --device cuda --dtype float32 --chunk-size 64 --sequence-length 64
python benchmark_qwen_sparse.py --model <local-qwen-snapshot> --tokenizer <local-qwen-snapshot> --local-files-only --device cuda --dtype float32 --chunk-size 32 --sequence-length 64
python benchmark_qwen_learned_sparse.py --model <local-qwen-snapshot> --tokenizer <local-qwen-snapshot> --local-files-only --device cuda --dtype float32 --chunk-size 256 --active-fraction 0.25 --router-steps 300
python benchmark_qwen_attention_transplant.py --model <local-qwen-snapshot> --tokenizer <local-qwen-snapshot> --local-files-only --device cuda --dtype float32 --layer-indices 14,27 --memory-size 256 --steps 300
```
