# V0.112 Qwen Routed Child Bank

## Question

Can a two-layer Qwen handoff use a bank of small learned attention-free
function circuits, activating only a top-k subset per token, instead of one
compact child block?

## Protocol

This extends the V0.109 two-layer protocol. Each child is a bank of four
`LayerNorm -> Linear(1024,192) -> GELU -> Linear(192,1024)` experts with a
small learned router. Training uses a soft mixture over all four experts;
evaluation uses hard top-k routing. The two children are trained sequentially
on layers 25 and 26, with eight train batches and four held-out batches.

```powershell
python -u benchmark_qwen_two_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --first-layer 25 --second-layer 26 --inner-size 192 --child-kind routed --num-experts 4 --active-experts 2 --routing-temperature 1.0 --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_two_layer_routed4x2_seed2026.json
python -u benchmark_qwen_two_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --first-layer 25 --second-layer 26 --inner-size 192 --child-kind routed --num-experts 4 --active-experts 2 --routing-temperature 1.0 --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2027 --output results/runs/qwen_two_layer_routed4x2_seed2027.json
python -u benchmark_qwen_two_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --first-layer 25 --second-layer 26 --inner-size 192 --child-kind routed --num-experts 4 --active-experts 3 --routing-temperature 1.0 --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_two_layer_routed4x3_seed2026.json
```

## Results

| experts | active | seed | alpha=0 two-child CE delta | teacher top-1 agreement | decision |
|---:|---:|---:|---:|---:|:---:|
| 4 | 2 | 2026 | +0.0544 | 94.14% | near-gate |
| 4 | 2 | 2027 | +0.0710 | 92.29% | fail |
| 4 | 3 | 2026 | +0.0766 | 93.65% | fail |

Each routed child stores 1,717,636 scalar parameters, while only two of four
expert bodies are selected at evaluation. The best seed is materially better
than the dense single-child handoff (+0.2377), but the result is not stable
across seeds and the extra active route did not improve quality.

## Decision

**The current routed child bank is not accepted as a deployable GO.** It is a
useful positive clue that functional specialization can reduce the two-layer
handoff error, but the soft-training/hard-routing transition remains unstable.
The route also computes all experts in this research implementation, so no
latency advantage is claimed.

Further Qwen routing tuning is paused. The next work follows `taklif22` and
tests whether reusable computation, mutable knowledge, working state, and
latent control can be separated on a controlled synthetic machine before
returning to language transfer.

## Artifact

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_routed4x2_seed2026.json`
- `results/runs/qwen_two_layer_routed4x2_seed2027.json`
- `results/runs/qwen_two_layer_routed4x3_seed2026.json`
