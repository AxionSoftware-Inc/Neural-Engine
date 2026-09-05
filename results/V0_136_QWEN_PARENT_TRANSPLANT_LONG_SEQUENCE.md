# V0.136 Qwen Parent Transplant on Longer Sequences

## Question

Does the positive single-layer Qwen-to-attention-free FFN handoff survive a
longer sequence and a second initialization seed?

## Protocol

Qwen3-0.6B is loaded locally in float32 on CUDA. The frozen late FFN at layer
26 is replaced by an attention-free
`LayerNorm -> Linear(1024,384) -> GELU -> Linear(384,1024)` child. The child
is trained for 600 steps on parent hidden-state input/output pairs using
sequence length 128, twice the V0.108 sequence length. The surrounding Qwen
Transformer remains frozen and unchanged. Evaluation sweeps the parent/child
mixture to alpha=0 and measures teacher CE delta and top-1 agreement.

```powershell
python -u benchmark_qwen_parent_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-index 26 --inner-size 384 --batch-size 4 --sequence-length 128 --steps 600 --learning-rate 0.003 --seed 2026 --log-every 100 --output results/runs/qwen_parent_transplant_layer26_longseq_seed2026.json
python -u benchmark_qwen_parent_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layer-index 26 --inner-size 384 --batch-size 4 --sequence-length 128 --steps 600 --learning-rate 0.003 --seed 2027 --log-every 100 --output results/runs/qwen_parent_transplant_layer26_longseq_seed2027.json
```

## Results

| seed | teacher CE | alpha=0 child CE delta | child top-1 agreement | local child MSE | decision |
|---:|---:|---:|---:|---:|:---:|
| 2026 | 1.1821 | **+0.0068** | **98.05%** | 2.2559 | pass |
| 2027 | 1.1821 | **+0.0171** | **98.44%** | 2.2392 | pass |

The child uses 789,888 scalar parameters versus 9,437,184 in the parent FFN,
or 8.37% of the parent size. The mean alpha=0 CE delta is `+0.0120`.

## Interpretation

The one-layer local transplant signal is stable across seeds and remains within
the V0.108 quality gate on a longer sequence. This supports turning an
individual Qwen FFN function into an attention-free Neural Engine circuit.

This does not validate replacing multiple adjacent layers or removing the
Transformer context: V0.109 and V0.112 still show representation drift in
two-layer handoffs. The result is therefore a local function-transfer GO, not
an end-to-end language-model replacement GO.

## Decision

**Keep the single-layer Qwen circuit-transfer path.** The next Qwen gate should
use the successful child as one member of a genuine sparse circuit bank and
measure hard top-k execution with teacher-logit fidelity. Do not scale to 1B+
or claim full attention removal until the multi-layer gate passes.

## Artifacts

- `benchmark_qwen_parent_transplant.py`
- `results/runs/qwen_parent_transplant_layer26_longseq_seed2026.json`
- `results/runs/qwen_parent_transplant_layer26_longseq_seed2027.json`
