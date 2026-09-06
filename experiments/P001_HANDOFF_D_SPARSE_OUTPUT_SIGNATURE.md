# Handoff D — sparse output-aware selection without dense bank inference

Branch: `exp/expert-handoff-d-sparse-output-signature`
Base: `exp/p001-retrieval-diagnostic` at `64f8381c84def4dc86fb79e2c106776f51ba1953`.
Scope: Handoff D only. Existing retriever, circuit bank and default model are unchanged.

## Starting signal

The frozen retrieval-window audit showed a large gap between the existing key
selector and a local output-aware proxy. At native M=8:

- seed17 same-key selector CE 2.5044 vs local proxy 2.3207;
- seed18 same-key selector CE 2.4634 vs local proxy 2.2237;
- local proxy hard accuracy was about 49.5–49.8%, versus about 47% for key score.

The proxy is not deployable because the diagnostic computes real circuit outputs
for the bank and uses the target label to rank pairs. Handoff D asks whether the
signal can be distilled into sparse inference instead of adopting the oracle.

## Operational ablation

The benchmark separates the proxy into the following controls on the same
frozen candidate states:

1. `key_score`: existing query-key selector; no output information.
2. `individual_output_additive`: each candidate is independently passed through
   the real immediate GRU + output head, then two individual losses are added.
   The gap from key score measures whether circuit-output awareness itself is
   useful before modelling pair interaction.
3. `joint_output_no_gru`: the two real candidate outputs are combined but the
   recurrent GRU transition is replaced by a residual update. The gap to the
   full local proxy measures immediate GRU contribution.
4. `joint_gru_state_norm`: real pair + real GRU, but selection uses state-change
   magnitude rather than the output head/label alignment. Its gap to the full
   local proxy is a head/target-alignment diagnostic.
5. `full_local_gru_head`: real candidate pair, real immediate GRU, real output
   head and target CE. This is the local teacher upper bound, not inference.
6. `sparse_signature`: the deployable student described below.

These ablations are not claimed to be a unique causal decomposition, but they
make the four requested ingredients observable without changing the circuit body.

## Sparse implementation: miniature output signatures

The single proposed inference implementation is `SparseOutputSignatureSelector`.
It does **not** change retrieval. The existing HierarchicalRouter still returns
M=8 candidate IDs.

For candidate circuit i and query q, the student computes a tiny learned
signature:

```text
h_i = GELU(q @ A_i)
z_i = h_i @ B_i + b_i
```

where `A_i` has rank `r_s` and `z_i` has `s` dimensions. Defaults are
`r_s=2`, `s=8`, much smaller than the real circuit body (`rank=8`, state=128 in
the canonical P-001 checkpoint).

Every pair among the eight signatures is scored by a small shared symmetric
head using:

- pair mean `(z_i + z_j)/2`;
- elementwise interaction `z_i * z_j`;
- absolute difference `|z_i-z_j|`;
- one shared query projection.

The winning pair is then executed by the original two real circuits. Pair
weights still come from the existing circuit keys and the existing tree
`route_gain` is preserved. Therefore the experiment changes selector choice,
not candidate retrieval, circuit implementation, recurrent state update or the
default model.

## Distillation target

Training freezes the full Neural Engine checkpoint. The selector sees training
labels only through a training-only teacher:

1. generate the current candidate pool M=8;
2. compute the real outputs of those eight candidates only;
3. evaluate all 28 candidate pairs with the real immediate GRU and output head;
4. obtain local final-class CE for every pair;
5. train the signature pair scores to match the teacher ranking/distribution.

The first 500 steps use native checkpoint states as a stable warmup. Afterwards
contexts are collected on-policy under the current sparse selector, so later
recurrent states reflect its own earlier pair choices.

The teacher is **candidate-only** even during training. It never computes all 32
real circuit outputs. Inference is stricter: it computes zero real candidate
outputs for scoring and executes only the selected two circuits.

## Sparse cost contract

Inference must report:

- exactly M signature rows touched per decision;
- exactly 2 real circuit rows executed per decision;
- 0 candidate real circuit rows used for scoring;
- 0 full-bank real circuit rows used for scoring;
- selector total/touched parameters and approximate multiply-adds;
- paired wall-clock latency against the unchanged checkpoint.

Dense full-bank scoring, even if accurate, fails the experiment by definition.

## Full gate

Full benchmark requires seed17/18, E=32, M=8, active=2, T=3 and 5000 selector
steps. PASS requires all of:

- mean held-out hard accuracy >= +2 pp, neither seed negative;
- mean candidate-selection regret improves >=10%;
- p95 candidate-selection regret improves >=10%;
- isolated frozen-state candidate recall does not decrease (retriever is unchanged);
- each seed inference latency <=1.25x control;
- cost report proves no dense full-bank or candidate real-output inference.

A local-proxy/oracle improvement alone is not sufficient. On failure,
`--update-problems-on-reject` appends a `REJECTED` record and keeps P-001 active.
A PASS does not automatically mark P-001 solved because retrieval itself can
remain a separate bottleneck.

## Reproduction

```bash
python -m pytest tests/test_p001_sparse_output_selector.py -q

python benchmark_p001_sparse_output_signature.py \
  --checkpoint <seed17-5000-step-checkpoint.pt> \
  --checkpoint <seed18-5000-step-checkpoint.pt> \
  --device cuda \
  --steps 5000 \
  --update-problems-on-reject
```

Smoke only:

```bash
python benchmark_p001_sparse_output_signature.py \
  --checkpoint <seed17-checkpoint.pt> \
  --device cpu \
  --smoke
```
