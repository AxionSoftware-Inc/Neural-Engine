# V0.184 Qwen contribution-diverse partition audit

Status: `REJECTED FOR ADOPTION`

## Hypothesis

V0.181 grouped similar output-space neuron signatures together.  This test
does the opposite: first form deterministic contribution clusters, then place
an equal slice of every cluster in every disjoint group.  Every active subset
therefore sees a broad mixture of functional directions.  The group width and
active compute remain unchanged.  V0.182 teacher-fitted signed subset
coefficients are used so this is a partition test rather than an `E/K` scale
test.

## Results

Qwen3-0.6B, float32 CUDA, layers 25--26, `E=8`, `K=4`, grouped dispatch,
four calibration batches, held-out `data/qwen_eval.txt`, and exact
best-subset routing under the signed reconstruction cost.

| Variant | Held-out +CE | Teacher top-1 | Local MSE | Gate |
|---|---:|---:|---:|:---:|
| Contribution-diverse exact oracle | `+0.0397` | `90.77%` | `0.781 / 0.951` | PASS |
| Contribution-diverse learned router, 100 steps | `+0.0697` | `88.82%` | `0.962 / 1.110` | FAIL |

The oracle improves the disjoint signed control (`+0.0374`) by only `+0.0023`
CE.  The learned router still misses the oracle and gives no adoption-quality
result.

## Decision

Reject contribution-diverse partition as a production/default architecture.
It is a useful negative control: changing group composition can move the
oracle slightly, but it does not produce the large scaling signal needed for
the Neural Engine goal.  No 4-layer or larger-model continuation is justified
from this small difference.  The remaining Qwen bottleneck is not solved by
simple similarity grouping, core overlap, or cluster-stratified diversity.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `tests/test_contribution_cluster.py`
- `results/runs/qwen_contribution_diverse_signed_oracle_2layers_seed2026.json`
- `results/runs/qwen_contribution_diverse_router_2layers_seed2026.json`

## Reproduction

```powershell
python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 4 --partition-mode contribution-diverse `
  --route-source oracle-subset --calibration-rank 1 `
  --calibration-mode signed-subset --dispatch-mode grouped `
  --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --train-batches 4 --eval-batches 2 `
  --batch-size 8 --sequence-length 128 --steps 0 --hard-train-steps 0 `
  --router-supervision-steps 0 --router-target subset --seed 2026 `
  --output results/runs/qwen_contribution_diverse_signed_oracle_2layers_seed2026.json
```
