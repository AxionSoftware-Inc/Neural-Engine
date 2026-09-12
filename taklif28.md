# taklif28.md — Knowledge-to-Weights Compiler: synthesize useful parameters from new information with minimal optimization

Status: **independent fast-learning proposal; no persistent model-size growth requirement in V0**

> **SCOPE:** Test whether an already trained model can transform new knowledge into adapter/expert parameters directly or with only a few refinement steps, instead of conventional long fine-tuning. This is distinct from selecting data (`taklif27`) and distinct from deciding whether total model size should grow.

## Hypothesis

A learned compiler/hypernetwork `G` can map a compact representation of new information `z(D)` into candidate parameters:

`phi_0 = G(z(D), theta_context)`

followed optionally by a short refinement:

`phi* = Refine(phi_0, D, k steps)`.

If successful, learning time is shifted from repeated gradient optimization of a large model to one forward parameter-synthesis pass plus small local correction.

## V0 experiment

Freeze a base model. Use fixed-size knowledge modules of equal parameter count. For many small held-out knowledge packs, compare:

1. random-init adapter + normal fine-tuning;
2. generic pretrained adapter + fine-tuning;
3. compiler-generated adapter with zero gradient steps;
4. compiler-generated adapter with `1/10/100` refinement steps.

The compiler must be trained on disjoint knowledge packs and tested on unseen topics/documents.

## Measurements

- time-to-target-quality;
- optimizer steps;
- GPU FLOPs;
- generated parameter norm/diversity;
- factual recall and paraphrase generalization;
- multi-hop use of learned knowledge;
- old-model regression;
- compiler inference cost.

## Success gate

A meaningful result requires reaching a fixed quality target substantially faster than ordinary adapter fine-tuning on unseen knowledge packs. The strongest result would be useful zero-shot parameter synthesis; a weaker but still valuable result is an order-of-magnitude reduction in refinement steps.

## Failure / stop rule

Reject if generated parameters merely reproduce a generic initialization, if the compiler only works on training-domain packs, or if its own cost approaches the cost of ordinary fine-tuning.

## Why it matters

This directly tests the user's strongest speed hypothesis: instead of asking a 27B model to relearn through long optimization, can a learned system **prepare the right new parameters** from the information itself?