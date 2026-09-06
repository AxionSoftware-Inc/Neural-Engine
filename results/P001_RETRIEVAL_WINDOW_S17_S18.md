# P-001 retrieval-window diagnostic

This is a frozen-bank counterfactual audit, not a trained-router result.
The native M=8 route state is held fixed while its contiguous candidate
window is widened. Pair quality is evaluated exhaustively at one changed
recurrent decision, with uniform weights and unit gain.

## Aggregate results

| Seed | M | Candidate oracle CE | Same-key selector CE | Full oracle CE | Retrieval regret | Selection regret | p95 retrieval | Recall | Candidate acc | Selector acc | Full acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 8 | 2.2420 | 2.5044 | 2.0467 | 0.1953 | 0.2624 | 0.9801 | 9.9% | 50.0% | 47.3% | 52.4% |
| 17 | 16 | 2.1566 | 2.5316 | 2.0467 | 0.1099 | 0.3750 | 0.5659 | 28.4% | 51.2% | 46.9% | 52.4% |
| 17 | 24 | 2.0849 | 2.5344 | 2.0467 | 0.0383 | 0.4495 | 0.1705 | 64.2% | 52.0% | 46.5% | 52.4% |
| 17 | 32 | 2.0467 | 2.5345 | 2.0467 | 0.0000 | 0.4878 | 0.0000 | 100.0% | 52.4% | 46.9% | 52.4% |
| 18 | 8 | 2.1875 | 2.4634 | 2.0165 | 0.1710 | 0.2759 | 0.8565 | 12.1% | 50.4% | 46.9% | 54.6% |
| 18 | 16 | 2.0910 | 2.5075 | 2.0165 | 0.0744 | 0.4165 | 0.3438 | 28.2% | 52.7% | 46.8% | 54.6% |
| 18 | 24 | 2.0543 | 2.5137 | 2.0165 | 0.0377 | 0.4595 | 0.2343 | 55.9% | 53.7% | 46.1% | 54.6% |
| 18 | 32 | 2.0165 | 2.5257 | 2.0165 | 0.0000 | 0.5092 | 0.0000 | 100.0% | 54.6% | 45.8% | 54.6% |

## Interpretation rule

If widening M sharply reduces retrieval regret/raises recall while the
same-key selector remains poor, retrieval is a real bottleneck but the
selector/objective still needs separate work. If M=16/24 barely changes
the result, candidate width is not the main explanation for scaling failure.

## Observed result and decision

The first condition is clearly present. Across both seeds, widening M from 8
to 16 reduced mean retrieval regret from `0.1953` to `0.1099` CE for seed17
and from `0.1710` to `0.0744` for seed18. M=24 reduced it further to `0.0383`
and `0.0377`; recall rose from `9.9%/12.1%` at M=8 to `64.2%/55.9%` at M=24.
At M=32 the full-bank oracle has zero retrieval regret by construction.

However, the same frozen key-score selector did not follow the final-loss
oracle. Its selection regret grew from `0.2624/0.2759` at M=8 to
`0.4878/0.5092` at M=32, while selector accuracy stayed around `46%–47%`.
The candidate oracle improved as M widened, but that improvement is only an
oracle upper bound; it is not an executable model result.

**Decision: diagnostic only; P-001 remains ACTIVE.** Candidate retrieval is a
real bottleneck, but increasing M alone is not an acceptable fix because the
existing selector/objective is not aligned with final corrected CE. The next
expert patch must preserve this decomposition and target retrieval plus
selection separately, or first provide a loss-aligned selector diagnostic.

Limitations: this is one-decision counterfactual evaluation on the two frozen
capacity checkpoints. Native M=8 route states, contiguous-window starts and
all other recurrent steps are held fixed. No new router was trained and no
claim about end-to-end scaling or latency follows from the oracle rows.

Exact JSON: `results/runs/p001_retrieval_window_s17_s18.json`.
