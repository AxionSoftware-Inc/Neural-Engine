# V0.183 Qwen deterministic core-overlap codebook audit

Status: `REJECTED`

## Hypothesis

Instead of a disjoint partition, repeat a high-energy core of Qwen FFN neurons
in every group and distribute the remaining lower-energy tail between groups.
The active compute and group width stay unchanged.  The signed-subset
reconstruction wrapper from V0.182 is used so duplicate core contributions can
be fitted with a proper subset-specific coefficient.

This is a deterministic control for an overlap/codebook idea.  It is distinct
from the earlier random overlap screen: the core is selected from calibration
output energy and the tail is interleaved deterministically.

## Protocol and results

Qwen3-0.6B, float32 CUDA, layers 25--26, `E=8`, `K=4`, grouped dispatch,
teacher-fitted signed subset coefficients, and exact best-subset routing under
the new reconstruction cost.  Because the oracle itself fails, a learned
router or deeper run would not be an informative continuation.

| Repeated core fraction | Held-out +CE | Teacher top-1 | Local MSE | Gate |
|---:|---:|---:|---:|:---:|
| 25% of each group | `+0.1429` | `84.18%` | `2.041 / 2.801` | FAIL |
| 12.5% of each group | `+0.1500` | `83.00%` | `2.260 / 3.288` | FAIL |

The disjoint signed-subset oracle from V0.182 was `+0.0374`; repeating the
core and dropping tail neurons therefore makes the reconstruction much worse.

## Decision

Reject this core-overlap codebook.  The failure occurs with an exact oracle,
so it is not a router-training problem.  Repeating high-energy neurons does
not compensate for the missing tail contributions in Qwen's nonlinear FFN.
The existing random-overlap result remains rejected as non-reproducible; this
deterministic overlap construction is now rejected on quality.  No 700M/1B
scale run is justified.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `tests/test_core_overlap.py`
- `results/runs/qwen_core_overlap_signed_oracle_2layers_seed2026.json`
- `results/runs/qwen_core_overlap12p5_signed_oracle_2layers_seed2026.json`

## Reproduction

```powershell
python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 4 --partition-mode core-overlap `
  --core-overlap-fraction 0.125 --route-source oracle-subset `
  --calibration-rank 1 --calibration-mode signed-subset `
  --dispatch-mode grouped --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --train-batches 4 --eval-batches 2 `
  --batch-size 8 --sequence-length 128 --steps 0 --hard-train-steps 0 `
  --router-supervision-steps 0 --router-target subset --seed 2026 `
  --output results/runs/qwen_core_overlap12p5_signed_oracle_2layers_seed2026.json
```
