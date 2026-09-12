# taklif26.md — Raw-Document Self-Learning: learn directly from MD/JSON/PDF/text without hand-made QA pairs

Status: **independent continual-learning proposal; persistent model growth is out of scope for V0**

> **SCOPE:** Test whether an already trained model can ingest ordinary documents and autonomously convert them into useful learning signals without a manually prepared question-answer fine-tuning dataset. V0 may update a small adapter/expert or memory layer, but must not also test structural growth.

## Hypothesis

Given raw material `D` (Markdown, JSON, extracted PDF text, code, documentation), the system can generate its own training curriculum and objectives:

`D -> parse -> concepts/facts/relations -> self-supervised tasks -> learning updates`.

Possible objectives include masked/reconstructed spans, next-section prediction, relation reconstruction, generated retrieval questions, consistency checks, and contrastive concept binding. The important condition is that the source material is ordinary information, not a human-authored QA dataset.

## V0 experiment

Start from a frozen base model and a small trainable knowledge adapter. Provide a corpus containing facts and relations absent from the base evaluation set. Compare:

1. no adaptation;
2. conventional supervised QA fine-tune built from the same source;
3. autonomous raw-document learning from the source only.

Use held-out questions, paraphrases, multi-hop combinations, and contradiction tests that were not generated verbatim during training.

## Measurements

- ingestion wall time and GPU time;
- tokens processed;
- trainable parameters touched;
- exact factual recall;
- paraphrase/generalization accuracy;
- multi-hop use of learned facts;
- old-capability regression;
- source-copying / memorization rate.

## Success gate

Raw-document learning should recover a substantial fraction of supervised-QA quality while requiring little or no manual dataset construction and without large regression on the base model.

## Failure / stop rule

Reject if the method only memorizes text spans, requires effectively generating a giant QA dataset first, or cannot use learned information under paraphrase/composition.

## Why it matters

This separates the question `can the model teach itself from ordinary information?` from the harder questions of where new knowledge should be stored and whether the model should grow new parameters.