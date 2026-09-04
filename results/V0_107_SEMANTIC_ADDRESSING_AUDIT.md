# V0.107 Self-Describing Semantic Addressing Audit

## Question

Can a small computational-need query retrieve a local candidate group and
active cells that reuse the same functional neighborhood across raw-value
changes, while descriptors remain tied to cell behavior?

## Protocol

This is a synthetic routing-only audit of the `taklif19` idea. There are 64
cells in eight fixed coarse groups, eight cells per group, and top-2 local
self-match. A query encoder receives one of three functional roles plus a
four-feature raw-value nuisance signal. The first three groups contain fixed
behavior signatures for the three roles; the remaining groups are distractors.

The coupled variant adds a descriptor-to-body behavior consistency loss. The
free variant uses the same route loss but no descriptor/body coupling. Both
variants use 1,500 steps and the same role/value distributions on CUDA.

```powershell
python -u experiment_semantic_addressing.py --device cuda --steps 1500 --output results/runs/semantic_addressing.json
```

## Results

| variant | coarse group accuracy | active cells match role | full-scan top-k recall | same-role route Jaccard | descriptor/body cosine |
|---|---:|---:|---:|---:|---:|
| coupled descriptor | **100.00%** | **100.00%** | **100.00%** | **1.00** | **0.729** |
| free descriptor | 100.00% | 100.00% | 100.00% | 1.00 | 0.011 |

The hierarchical route evaluates 16 dot products (eight coarse plus eight
local) instead of 64 for a full scan. Both variants learn the role route, but
only the coupled variant keeps the searchable descriptor aligned with the
fixed body behavior. The free control confirms that route accuracy alone is
not evidence of a self-describing cell.

## Decision

The semantic-address mechanism passes its synthetic routing audit and remains
worth testing with actual learned cell bodies. This is not a Neural Engine
quality result: cell behavior was fixed and role labels were available. The
next gate must use a variable-depth task with learned bodies and audit
descriptor/body consistency, route reuse under value changes, and
counterfactual route penalties. Do not scale the bank before that gate.

## Artifact

- `experiment_semantic_addressing.py`
- `results/runs/semantic_addressing.json`
