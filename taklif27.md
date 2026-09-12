# taklif27.md — Novelty-Gated Selective Learning: learn only what the model does not already know

Status: **independent data-efficiency proposal; no structural parameter growth in V0**

> **SCOPE:** Test whether an existing model can reduce continual-learning cost by deciding which incoming information is novel, useful, contradictory, or redundant before applying gradient updates. This proposal is not about raw-document curriculum generation itself and not about adding new parameters.

## Hypothesis

Most incoming corpora contain large amounts of information the model already represents or that adds little useful capability. Instead of training on every token uniformly, compute a learning utility score:

`U(d) = novelty + uncertainty + contradiction + expected_utility - redundancy - learning_cost`.

Only items above a threshold are admitted to training, while redundant material is skipped or stored externally.

## V0 experiment

Use a base model plus one fixed-size trainable adapter. Build an incoming corpus with known proportions of:

- already-known facts;
- paraphrases/duplicates;
- genuinely new facts;
- new relations requiring composition;
- contradictions/noisy claims.

Compare full-corpus continual training against selective admission at several retained-data fractions such as `100%, 50%, 20%, 10%, 5%`.

## Measurements

- total tokens trained on;
- GPU time and wall time;
- number of optimizer steps;
- new-knowledge quality;
- old-capability regression;
- false-reject rate for useful new knowledge;
- false-admit rate for redundant/noisy knowledge;
- quality per training FLOP.

## Success gate

A strong positive result is near-full new-knowledge quality using a substantially smaller retained fraction and substantially less training compute. Example target: retain <=20% of incoming material while preserving >=95% of the quality gain of full-corpus adaptation.

## Failure / stop rule

Reject if utility scoring costs as much as training, if useful information is routinely filtered out, or if the retained subset gives materially worse generalization than full-corpus learning.

## Why it matters

This proposal directly tests whether continual learning can be much faster because the model does not relearn what it already knows. The possible speedup comes from **less optimization work**, not from assuming that data selection alone creates new knowledge for free.