# P-001 minimal retrieval patch — strided M=8 candidate schedule

**Status: REJECTED FOR ADOPTION (acceptance evidence incomplete; P-001 remains ACTIVE)**

This note records the Handoff A decomposition, the single opt-in patch that was
added, and why it is not promoted. No result below is invented from an
unavailable checkpoint.

## 1. Retrieval and selection are separate failures

The existing frozen-bank P-001 audit already establishes both components.

| Seed | M | Recall | Retrieval regret mean | Retrieval p95 | Selection regret mean | Same-key selector CE | Selector acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 8 | 9.9% | 0.1953 | 0.9801 | 0.2624 | 2.5044 | 47.3% |
| 17 | 16 | 28.4% | 0.1099 | 0.5659 | 0.3750 | 2.5316 | 46.9% |
| 17 | 24 | 64.2% | 0.0383 | 0.1705 | 0.4495 | 2.5344 | 46.5% |
| 18 | 8 | 12.1% | 0.1710 | 0.8565 | 0.2759 | 2.4634 | 46.9% |
| 18 | 16 | 28.2% | 0.0744 | 0.3438 | 0.4165 | 2.5075 | 46.8% |
| 18 | 24 | 55.9% | 0.0377 | 0.2343 | 0.4595 | 2.5137 | 46.1% |

M=8 -> M=16 reduces retrieval p95 by about 42.3% on seed17 and 59.9% on
seed18, while recall rises by +18.5 pp / +16.1 pp. At the same time selection
regret increases by about 42.9% / 51.0%, selector CE becomes worse by
+0.0272 / +0.0441, and selector accuracy does not improve. Therefore:

- retrieval is a real bottleneck;
- widening M is not an acceptable solution;
- the existing key-score selector is not aligned with final corrected CE;
- retrieval and selection must remain separate metrics.

The patch below deliberately does not change the selector because Handoff A
requires the first P-001 patch to isolate candidate retrieval.

## 2. Single minimal opt-in patch

Added:

- `neural_engine/p001_retrieval_patch.py`
- `tests/test_p001_retrieval_patch.py`
- `benchmark_p001_strided_retrieval.py`

The experiment keeps E=32, M=8, active=2 and T=3. Instead of one contiguous
8-circuit candidate window, it uses the router's already-existing
`routing_windows` interface to expose four 2-circuit windows spread over the
32-circuit bank. In the canonical case the windows start at 0, 8, 16 and 24 and
share the same local leaf coordinate.

It is runtime opt-in through a removable forward pre-hook. The normal
`HierarchicalRouter` constructor, checkpoint state dict and default model path
are untouched.

Not changed:

- pair/key-score selector;
- circuit body;
- recurrent state update;
- route weights or route gain;
- correction path;
- default model;
- Attention/Transformer use (none added).

No trainable parameter is added. Candidate-key reads remain 8 rows per decision
and circuit execution remains 2 rows per decision. Training probe extra
forwards and training probe parameters are both zero. Runtime latency is still
measured separately by the paired benchmark rather than assumed to be free.

## 3. Paired seed17/18 benchmark contract

`benchmark_p001_strided_retrieval.py` measures, on identical held-out streams:

### Quality

- held-out CE;
- hard accuracy;
- dead circuits;
- paired inference seconds/example and latency ratio.

### Retrieval / selection decomposition

- candidate recall of the full-bank oracle pair;
- candidate-oracle CE;
- full-bank-oracle CE;
- retrieval regret mean and p95;
- existing same-key selector CE/accuracy;
- selection regret mean and p95;
- per-internal-step versions of the same metrics.

### Active/training cost

- candidate key rows and scalar key reads per decision/example;
- tree score multiply-add estimate;
- candidate score multiply-add estimate;
- executed circuit rows per decision/example;
- `parameter_report()` active/total fields when available;
- patch trainable parameters = 0;
- extra training probe forwards = 0.

The gate is fail-closed and requires all of the following:

1. seed17 and seed18, E=32, active=2, M=8, T=3;
2. source checkpoints report at least 5000 training steps;
3. candidate recall does not decrease;
4. mean p95 retrieval regret improves by at least 10%;
5. mean p95 selection regret improves by at least 10%;
6. mean held-out hard accuracy improves by at least +2 pp;
7. dead circuits <=3/32;
8. latency <=1.25x;
9. active-cost accounting is unchanged structurally.

CE is reported but is never sufficient by itself.

## 4. Execution status in this handoff

The seed17/18 checkpoint files used by the historical audits are intentionally
local/ignored and are not present in this branch under `results/checkpoints/`.
The earlier P-001 commit also references
`results/runs/p001_retrieval_window_s17_s18.json`, but that JSON was not
committed; only the aggregate Markdown is available. The branch has no GitHub
Actions run or release artifact containing the checkpoints.

The execution container available for this handoff also cannot resolve
`github.com`, so it cannot clone the repository independently and regenerate a
5000-step CUDA checkpoint from the connected source.

Because the required paired seed17/18 patch metrics (hard accuracy, CE, p95
selection/retrieval regret, dead circuits and latency) cannot be produced from
the committed artifacts, the acceptance gate is **not satisfied**. Per Handoff
A's fail-closed rule, this patch is therefore **REJECTED FOR ADOPTION**. This is
not a claim that the strided schedule hypothesis is scientifically disproven;
it means it has not earned promotion and must not become default.

P-001 remains `ACTIVE`.

## 5. Exact reproduction commands

Small paired smoke on the real local seed17/18 checkpoints:

```bash
python benchmark_p001_strided_retrieval.py \
  --checkpoint /path/to/seed17.pt \
  --checkpoint /path/to/seed18.pt \
  --device cuda \
  --quality-batches 1 \
  --quality-examples-per-task 4 \
  --diagnostic-batches 1 \
  --diagnostic-examples-per-task 2 \
  --output results/runs/p001_strided_retrieval_smoke.json \
  --markdown results/P001_STRIDED_RETRIEVAL_SMOKE.md
```

Acceptance run:

```bash
python benchmark_p001_strided_retrieval.py \
  --checkpoint /path/to/seed17.pt \
  --checkpoint /path/to/seed18.pt \
  --device cuda \
  --quality-batches 8 \
  --quality-examples-per-task 32 \
  --diagnostic-batches 2 \
  --diagnostic-examples-per-task 17 \
  --output results/runs/p001_strided_retrieval_s17_s18.json \
  --markdown results/P001_STRIDED_RETRIEVAL_S17_S18.md
```

If that command produces `gate.decision == "REJECTED"`, retain the generated
JSON/Markdown as the final negative result and do not modify the default model.
If it produces `PASS`, it is still only the P-001 retrieval patch result; P-005
selector/objective work remains a separate handoff.
