# P-001 retrieval-window diagnostic

This is a frozen-bank counterfactual audit, not a trained-router result.
The native M=8 route state is held fixed while its contiguous candidate
window is widened. Pair quality is evaluated exhaustively at one changed
recurrent decision, with uniform weights and unit gain.

## Aggregate results

| Seed | M | Candidate oracle CE | Same-key selector CE | Local proxy final CE | Full oracle CE | Retrieval regret | Selection regret | Local proxy regret | Recall | Selector acc | Proxy acc | Full acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 8 | 2.2420 | 2.5044 | 2.3207 | 2.0467 | 0.1953 | 0.2624 | 0.0787 | 9.9% | 47.3% | 49.5% | 52.4% |
| 17 | 16 | 2.1566 | 2.5316 | 2.2482 | 2.0467 | 0.1099 | 0.3750 | 0.0917 | 28.4% | 46.9% | 50.3% | 52.4% |
| 17 | 24 | 2.0849 | 2.5344 | 2.1998 | 2.0467 | 0.0383 | 0.4495 | 0.1149 | 64.2% | 46.5% | 50.9% | 52.4% |
| 17 | 32 | 2.0467 | 2.5345 | 2.1648 | 2.0467 | 0.0000 | 0.4878 | 0.1181 | 100.0% | 46.9% | 51.3% | 52.4% |
| 18 | 8 | 2.1875 | 2.4634 | 2.2237 | 2.0165 | 0.1710 | 0.2759 | 0.0362 | 12.1% | 46.9% | 49.8% | 54.6% |
| 18 | 16 | 2.0910 | 2.5075 | 2.1406 | 2.0165 | 0.0744 | 0.4165 | 0.0497 | 28.2% | 46.8% | 52.0% | 54.6% |
| 18 | 24 | 2.0543 | 2.5137 | 2.1114 | 2.0165 | 0.0377 | 0.4595 | 0.0571 | 55.9% | 46.1% | 52.2% | 54.6% |
| 18 | 32 | 2.0165 | 2.5257 | 2.0762 | 2.0165 | 0.0000 | 0.5092 | 0.0597 | 100.0% | 45.8% | 53.4% | 54.6% |

## Interpretation rule

If widening M sharply reduces retrieval regret/raises recall while the
same-key selector remains poor, retrieval is a real bottleneck but the
selector/objective still needs separate work. If M=16/24 barely changes
the result, candidate width is not the main explanation for scaling failure.

The local proxy uses each frozen circuit output, the real GRU update and
the immediate output head, but omits suffix rerouting. It is a cheap
mathematical engineering probe, not an end-to-end model result.
If its final selected loss approaches the candidate oracle, a local
output-aware selector is promising; otherwise the selector likely needs
cascade-aware cost prediction.

Exact JSON: `results/runs/p001_retrieval_window_s17_s18.json`.
