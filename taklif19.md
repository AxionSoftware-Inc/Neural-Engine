# taklif19.md — Self-describing MacroCells, semantic addressing, sparse execution, and full falsification plan

Status: **canonical research specification; not yet validated**

This proposal formalizes the MacroCell idea as a falsifiable architecture rather than an open-ended architecture search. It is intentionally detailed because the main failure mode is not only implementation error; routing, representation, specialization, training dynamics, compute overhead, and evaluation leakage can each create a false positive or false negative.

---

# 0. Core hypothesis

The current project has already shown that very small active computation can work on a narrow structured task, but pretrained dense Qwen FFN chunks do not become useful 4x sparse primitives merely by routing a subset of them. The central hypothesis here is therefore not “build a better router over the same tiny chunks”. It is:

> Dense/distributed computation can be re-expressed as a small set of reusable, stronger, self-describing computational cells whose semantic addresses are cheap to retrieve and whose active subset remains small.

Canonical abstraction:

```text
state / working memory
        ↓
need encoder
        ↓
semantic address
        ↓
small candidate set
        ↓
candidate self-match
        ↓
Top-k MacroCells
        ↓
READ → COMPUTE → WRITE
        ↓
updated state
        ↓
repeat / halt
```

The architecture is a failure if useful quality requires a large fraction of MacroCells active, if routing cost scales linearly with bank size, if cells do not specialize, or if descriptors drift away from actual behavior.

---

# 1. Objects and notation

Let:

- `x_t ∈ R^d` = current token/local state at computation step `t`;
- `m_t ∈ R^(S×d_m)` = persistent working-memory/register state with `S` slots;
- `N` = number of stored MacroCells;
- `k << N` = number of active cells per step;
- `q_t ∈ R^p` = computational-need query;
- `κ_i ∈ R^p` = semantic capability signature/key of MacroCell `i`;
- `F_i` = actual computation body of MacroCell `i`;
- `a_i(q_t, x_t, m_t)` = cheap candidate self-match/acceptance score;
- `R(q_t)` = candidate retrieval function;
- `A_t ⊂ {1,…,N}` = final active cell set.

Each cell is represented as:

`M_i = (κ_i, F_i, optional metadata/state)`.

Important: `κ_i` must not be treated as a free semantic label with no relationship to `F_i`. Descriptor/body consistency is a first-class training and audit requirement.

---

# 2. MacroCell V1 — deliberately constrained

Do not start with attention, recursion, inner routers, long internal programs, convolutions, or arbitrary plugin-like operators. The first canonical cell must be powerful enough to subsume the current low-rank micro-circuit while still being cheap and analyzable.

For one selected memory summary `r_i(m_t)` and local state `x_t`:

`u_i = W_x,i x_t + W_m,i r_i(m_t) + W_b,i (P_x,i x_t ⊙ P_m,i r_i(m_t)) + b_i`

`h_i = φ(u_i)`

`y_i = W_o,i h_i`

`w_i = W_w,i h_i`

Then a selected cell emits both a state contribution and a memory-write proposal:

`F_i(x_t,m_t) = (y_i, w_i)`.

Minimal V1 properties:

1. linear input transform;
2. memory/register read;
3. multiplicative/bilinear interaction;
4. one nonlinear latent transform;
5. output residual;
6. memory-write proposal.

No internal attention in V1.
No internal dynamic routing in V1.
No more than one explicit nonlinear hidden stage in the first experiment.

The purpose is to test the MacroCell hypothesis, not hide failure by making each cell an entire Transformer block.

---

# 3. Expressivity relation to current micro-circuits

Current simplified micro-circuit form:

`C_i(x) = U_i φ(D_i x) + b_i`.

MacroCell V1 can recover this class by setting the memory and bilinear branches to zero and choosing:

- `W_x,i = D_i`;
- `W_o,i = U_i`;
- memory write disabled;
- compatible bias placement.

Therefore the old micro-circuit function class is embedded in the MacroCell class:

`C_micro ⊆ C_macro`.

This is an expressivity inclusion, not a proof of better trainability or better sparsity.

Likewise, if a MacroCell hidden dimension/depth is large enough, a sequence of several old transforms can be represented/approximated by one cell. That only establishes representational capacity. It does **not** establish that SGD will discover this compression, that semantic routing will be stable, or that a small active set will preserve real language quality.

---

# 4. Need encoder

The router should not ask “which of one million cells is best?” by scanning all cells. It should ask “what kind of computation is needed next?”

Define:

`q_t = Q(x_t, summary(m_t), optional_context_t)`.

`Q` must be much smaller than the total stored bank and should not scale linearly with `N`.

The query is not required to be human-interpretable. It is a learned computational address.

A successful architecture should show that similar functional needs produce nearby queries even when raw operands/content differ.

Required audit:

`same functional role + different values/content → high query/address reuse`.

If queries are dominated by raw values/token identity and route to different cells for every surface form, the architecture is memorizing addresses rather than learning reusable computation.

---

# 5. Hierarchical semantic addressing

A full scan:

`score_i = q_tᵀ κ_i,  i=1…N`

costs `O(Np)` and is rejected as the production design.

Use a hierarchical address tree/product code/learned partition. Canonical V1 example:

```text
Level 1: B1 coarse addresses
Level 2: B2 sub-addresses inside selected coarse address
Leaf: L MacroCell candidates
```

Retrieval cost approximately:

`C_route ≈ O(B1 p + B2 p + L p)`

rather than `O(Np)`.

For example, the system may expose a very large bank while evaluating only a few dozen/hundred small address scores. Exact numbers are experimental, not fixed claims.

Required scaling property:

`d C_route / d N` should be sublinear and preferably logarithmic/near-constant over the tested range.

If doubling stored cells approximately doubles route cost, the routing architecture fails the project objective even if accuracy is high.

---

# 6. Candidate self-match

The global addresser should only reduce the search to a small candidate set:

`C_t = R(q_t)`, where `|C_t| = L` and `L` is small.

Each candidate cell may then emit a cheap compatibility score:

`a_i = g_i(q_t, x_t, summary(m_t))` for `i ∈ C_t`.

Final active set:

`A_t = TopK({a_i : i ∈ C_t}, k)`.

Only `L` candidates self-match. All `N` cells must **not** wake up and vote.

This gives:

```text
cheap global addressing
        ↓
small candidate set
        ↓
local self-recognition
        ↓
expensive body execution only for Top-k
```

Potential failure: `g_i` becomes so expensive that routing savings disappear. Therefore self-match parameter count and MACs must be logged separately from cell-body compute.

---

# 7. What “self-describing cell” means

A MacroCell does not literally understand itself. “Self-describing” means its searchable capability descriptor is constrained to correspond to actual observed behavior.

Bad design:

`κ_i` is a free learned vector with no coupling to `F_i`.

This can produce descriptor/body mismatch:

```text
key says role A
body actually performs role B
```

Three progressively stronger descriptor mechanisms should be considered.

## 7.1 Usage-derived prototype

Maintain an EMA of successful queries:

`p_i ← (1-β) p_i + β E[q_t | cell i useful/selected with positive advantage]`.

Then either set:

`κ_i = normalize(p_i)`

or regularize learned `κ_i` toward `p_i`.

Interpretation: the cell’s address describes the computational needs for which it has historically been useful.

## 7.2 Behavioral probe signature

Use fixed/procedurally refreshed probe inputs `P={z_1,…,z_J}`.

Compute a compressed behavioral fingerprint:

`b_i = H(F_i(z_1),…,F_i(z_J))`.

Map to address space:

`κ̂_i = E_b(b_i)`.

Consistency loss:

`L_desc = ||normalize(κ_i) - stopgrad(normalize(κ̂_i))||²`

or a symmetric/EMA variant.

Risk: fixed probes may miss important behavior and can themselves become an overfit target. Therefore probe sets must include held-out/procedural variants.

## 7.3 Functional-advantage signature

For candidate cell `i`, estimate its marginal utility on sampled examples:

`Δ_i = L_without_i - L_with_i`.

Positive-advantage examples define the semantic region that cell should claim. This is expensive and is mainly an audit/teacher signal, not a runtime mechanism.

---

# 8. Descriptor/body consistency metrics

Do not trust qualitative route visualizations. Measure:

1. **behavior–key correlation**: key similarity vs functional-output similarity;
2. **usage–key correlation**: cells with nearby keys should serve related queries;
3. **counterfactual route penalty**: forcing a query to semantically distant cells should hurt more than forcing nearby cells;
4. **descriptor drift** over training;
5. **semantic reuse**: same functional role across content/value changes should return same neighborhood;
6. **collision rate**: unrelated behaviors sharing one address region;
7. **duplicate rate**: many cells with nearly identical behavior.

A high accuracy model with low descriptor/body consistency does not validate semantic addressing.

---

# 9. Sparse execution equations

Given selected cells `A_t` and normalized active weights `α_i`:

Parallel form:

`Δx_t = Σ_{i∈A_t} α_i y_i`

`x_{t+1} = x_t + Δx_t`

Memory proposal aggregation:

`Δm_t = Aggregate({α_i w_i})`

`m_{t+1} = Update(m_t, Δm_t)`.

Serial/compositional form:

`(x^(0),m^(0)) = (x_t,m_t)`

`(δx_j,δm_j)=F_{i_j}(x^(j-1),m^(j-1))`

`x^(j)=x^(j-1)+α_j δx_j`

`m^(j)=Update(m^(j-1),α_j δm_j)`

for `j=1…k`.

Serial mode is more expressive as a micro-program but more latency-sensitive and harder to parallelize. Both must be benchmarked; do not assume serial is superior because it is conceptually elegant.

---

# 10. Compute objective

Total per-step cost:

`C_total = C_need + C_route + C_selfmatch + k·C_cell + C_memory + C_dispatch`.

Stored capacity:

`P_stored = P_controller + Σ_i P_cell,i + P_address`.

Active parameters are not enough. Report:

- active parameter count;
- actual MAC/FLOP proxy;
- bytes read from memory;
- routing MACs;
- gather/dispatch overhead;
- latency;
- throughput;
- peak VRAM/RAM;
- cache hit rate when relevant.

Necessary inequality for the design to be useful:

`C_need + C_route + C_selfmatch + C_dispatch << cost saved by not executing inactive cells`.

If sparse execution is slower than a dense baseline, the architecture has not achieved a systems win even if active parameter count is tiny.

`taklif18` is the required systems companion for grouped/fused execution.

---

# 11. Relation to factorized virtual capacity

MacroCells and factorized virtual circuits are independent hypotheses. Do not combine them in the first experiment.

First test **physical MacroCells** directly.

Only if MacroCells show useful specialization/sparsity should a factorized version be tested:

`shared basis factors + cell-specific coefficients → virtual MacroCell`.

Potential benefit:

- larger stored/virtual functional space;
- gradient sharing;
- cheaper growth;
- fewer independent islands.

Potential failure:

- factor sharing makes cells too similar;
- descriptor regions collapse;
- virtual capacity becomes nominal rather than functional;
- physical parameter growth, not virtual address count, explains gains.

Always report separately:

`physical trainable params`, `virtual capacity`, `active params`, `active cells`.

---

# 12. Training chicken-and-egg problem

At initialization:

- cells do not yet have specialties;
- router does not know which cell should receive which examples;
- descriptors do not yet have semantic meaning.

If hard top-k routing is enabled immediately, random early routes can create a self-reinforcing monopoly:

```text
random route
→ only selected cell gets gradient
→ selected cell improves
→ router selects it more
→ other cells die
```

This can mimic “specialization” while actually being path dependence/collapse.

Therefore training is staged.

---

# 13. Training Stage A — dense/soft capability formation

Start with a small bank or broad candidate activation. Use soft weights over a moderate candidate set:

`π_i = softmax(s_i/τ)`

`y = Σ_i π_i F_i(x,m)`.

The purpose is not production sparsity. The purpose is to allow many cells to receive useful gradients and discover functional niches.

Temperature `τ` should not be annealed aggressively before cells have measurable differentiation.

Required logs:

- per-cell gradient frequency;
- route entropy;
- output similarity across cells;
- utilization distribution;
- dead-cell fraction.

---

# 14. Training Stage B — specialization and query/key alignment

For cells that demonstrate positive marginal utility, align the need query to their descriptors.

Contrastive routing objective:

`L_route = -log exp(sim(q,κ_+)/τ) / Σ_{j∈C} exp(sim(q,κ_j)/τ)`.

Positive labels can come from:

- teacher/oracle contribution;
- marginal loss improvement;
- soft-route posterior responsibility;
- stable usage statistics.

Do not assume current router choice is ground truth; otherwise the model merely reinforces its own mistakes.

---

# 15. Training Stage C — sparsity annealing

Gradually reduce active budget:

```text
broad soft set
→ 32
→ 16
→ 8
→ 4
→ 2
```

Exact schedule depends on bank size and task.

At every stage record the quality/active-compute curve.

Do not continue sparsifying after the pre-declared quality gate fails.

Hard routing training options:

- straight-through top-k;
- Gumbel/relaxed top-k;
- teacher-labeled routing;
- local soft candidate set + hard inference.

All introduce estimator bias/variance. Compare at least one alternative if optimization failure is suspected.

---

# 16. Training Stage D — global end-to-end fidelity

Local cell-output imitation is insufficient because small errors accumulate across layers/steps.

For pretrained-transfer experiments use:

`L_local = ||student_block(x)-teacher_block(x)||²`

`L_KL = KL(p_teacher || p_student)`

`L_LM = next-token/task loss`.

Combined core objective:

`L_core = λ_LM L_LM + λ_KL L_KL + λ_local L_local + λ_route L_route`.

Only add extra regularizers when a measured failure requires them.

Do **not** start with seven simultaneously tuned loss coefficients; that creates a hyperparameter swamp and makes causality impossible to interpret.

---

# 17. Optional regularizers — only when justified

## 17.1 Descriptor consistency

`L_desc` as defined above.

Use if descriptors fail to reflect behavior.

## 17.2 Stability

`L_stable = Σ_i ||κ_i^t - stopgrad(EMA(κ_i))||²`.

Use only after initial specialization. Strong early stability can freeze random initial semantics.

## 17.3 Diversity

Naive key repulsion:

`Σ_{i≠j} sim(κ_i,κ_j)`

can be harmful because related skills should legitimately cluster.

Prefer local anti-collapse penalties or behavior-aware diversity rather than forcing all cells orthogonal.

## 17.4 Load balance

Do not force uniform usage. Some primitives are naturally common.

Penalize only pathological monopoly/dead-cell collapse.

## 17.5 Capacity/entropy control

Routing entropy should decrease as specialization develops, but overly low entropy early can create irreversible dead paths.

---

# 18. Major failure modes

## F1 — cell collapse

Many cells learn the same general-purpose transform.

Symptoms:

- high behavioral similarity;
- low marginal benefit from more cells;
- arbitrary route swaps do not hurt.

Response:

- improve responsibility signal;
- mild behavior-aware diversity;
- increase task heterogeneity;
- do not simply increase cell count.

## F2 — route monopoly / dead cells

A few cells receive almost all traffic.

Distinguish useful common primitives from pathological monopoly using counterfactual and marginal-utility audits.

## F3 — descriptor/body mismatch

Keys look organized but do not predict actual functional behavior.

This invalidates “self-describing” claims even if accuracy is high.

## F4 — semantic drift

Cell role changes faster than router/index can track.

Possible response: EMA descriptors, periodic reindexing, late-stage stabilization.

## F5 — over-specialization/memorization

Cells specialize to raw values/tokens/examples rather than reusable operations.

Audit route reuse under value/content perturbations and strict OOD splits.

## F6 — distributed-knowledge persistence

Even strong MacroCells may require dozens/hundreds active for teacher-level fidelity.

If an oracle needs a large active fraction, better routing cannot solve the representation problem.

This is the most important falsification criterion.

## F7 — routing cost dominates

A beautiful sparse model can be slower because of dynamic indexing/gather/dispatch.

Measure real hardware, not only active params.

## F8 — MacroCell becomes a hidden Transformer expert

If each cell becomes huge/attention-heavy, low active-cell count can be meaningless because active compute remains large.

Always compare active MACs and memory traffic, not cell count.

## F9 — optimization instability

Router, descriptors, bodies, and memory all move simultaneously.

Symptoms: route churn, oscillating utilization, non-reproducible seeds.

Use staged training and freeze/unfreeze controls.

## F10 — benchmark saturation

A 99%+ narrow task can hide lack of generality. Use `taklif17` capability-frontier/OOD protocol.

## F11 — teacher-data overfit

Small calibration/evaluation sets can show negative CE deltas while teacher top-1/KL fidelity is poor.

Use independent train/calibration/test corpora and larger held-out controls.

## F12 — parameter-count illusion

Virtual capacity or large stored bank may not produce new functional capacity.

Measure capability frontier and functional diversity, not only nominal address count.

---

# 19. Mathematical properties that can and cannot be claimed

Can be established analytically/structurally:

1. MacroCell V1 contains the old micro-circuit as a special case.
2. Hierarchical routing can have sublinear lookup complexity if the index structure is fixed/learned appropriately.
3. Only `k` expensive bodies need execution after candidate retrieval.

Cannot be proven from architecture alone:

1. SGD will discover useful specialization.
2. keys will align with behavior.
3. 2–4 active cells can preserve Qwen quality.
4. larger stored cell count will improve capability.
5. sparse execution will be faster on real hardware.
6. OOD/general reasoning will emerge.

These are empirical hypotheses and require controlled experiments.

---

# 20. First falsification experiment — do not rebuild the whole NE

Teacher: real pretrained Qwen3-0.6B.

Target only a small set of late FFNs first (recommended late 4–6 layers) because current experiments show late layers are a cheaper controlled transfer surface than full-stack replacement.

Compare:

A. original dense Qwen FFN;
B. current contiguous Qwen chunks at matched active fraction;
C. compact dense student at matched compute/parameter budget;
D. MacroCell bank with semantic routing;
E. random routing control;
F. oracle MacroCell selection upper bound if feasible.

Initial MacroCell setup should be small enough to train/audit thoroughly, for example 8–32 cells per replaced block/family, not thousands.

Test active budgets such as:

`100% → 50% → 25% → 12.5%`, with exact cell counts reported.

Primary question:

> Does a learned MacroCell representation move the quality-vs-active-compute frontier materially beyond contiguous Qwen chunks and a compact dense student?

If no, stop before scaling bank size.

---

# 21. Required metrics for the Qwen pilot

Quality/fidelity:

- teacher/student CE on independent text;
- KL divergence;
- teacher top-1 agreement;
- logit MSE;
- block-output MSE/cosine;
- perplexity delta;
- later: standard language/reasoning benchmarks.

Routing/representation:

- active fraction;
- candidate recall against oracle;
- route entropy;
- utilization distribution;
- dead-cell fraction;
- query reuse under content/value perturbation;
- descriptor/body correlation;
- route churn across checkpoints;
- counterfactual route penalty;
- functional diversity.

Systems:

- physical params;
- active params;
- active cells;
- route MACs;
- cell MACs;
- bytes moved;
- latency;
- throughput;
- peak VRAM;
- dispatch/gather overhead.

---

# 22. GO / NO-GO gates

These are research gates, not sacred universal thresholds, but they must be declared before large scaling.

## Gate A — representation advantage

GO only if MacroCells clearly beat contiguous Qwen chunks at matched active compute and training budget.

If they merely tie, the extra architecture complexity is not justified.

## Gate B — compact dense control

MacroCells should beat or provide a distinct systems advantage over a simple compact dense FFN at matched compute. Otherwise the result may just show that retraining helps, not that sparse semantic cells help.

## Gate C — useful sparsity

Strong target:

`~25% active cell-body compute` with small teacher-fidelity loss.

A 2x reduction may still be useful, but if only ~80–100% of cells are needed for fidelity, the central sparse-representation hypothesis fails.

## Gate D — semantic organization

Same functional need across varying content should repeatedly address the same neighborhood. If routing is example-ID/value dominated, NO-GO for semantic addressing.

## Gate E — real speed path

No production speed claim until grouped/fused implementation is faster than matched dense execution on real hardware.

## Gate F — seed stability

At least two independent seeds before interpreting a positive routing/specialization result as architectural evidence.

---

# 23. Second experiment — synthetic capability frontier

Only after the small Qwen/local transfer gate or in parallel as a cheap architecture audit.

Use variable-length multi-step programs with explicit working memory:

```text
read operands
→ choose operation
→ write intermediate
→ choose next computation
→ branch/continue
→ output
```

Tasks must include:

- 2/4/6/8+ step compositions;
- variable program graphs;
- held-out operation combinations;
- unseen operand/content ranges;
- distractor state;
- dynamic memory-slot selection.

Compare bank sizes while controlling physical params, active compute, training steps/tokens, and seeds.

Desired evidence:

`larger stored semantic bank → harder frontier solved while k remains roughly fixed`.

A saturated 99.6% fixed two-step task is insufficient.

---

# 24. Growth experiment for MacroCells

Do not call architecture-changing transfer “parent growth” unless the mapping is specified.

Within the same MacroCell architecture, growth can be:

1. copy existing cells/descriptors/controller;
2. add new initialized cells/address leaves;
3. preserve old route compatibility where possible;
4. continue training.

Controls:

- parent continued for equal additional compute without growth;
- grown child with same cumulative compute;
- scratch child with matched total compute if feasible.

Without equal-compute controls, a grown model outperforming scratch does not prove added capacity caused the gain.

Transformer→MacroCell transfer belongs to `taklif16` cross-architecture transplant and should use teacher/adapter/progressive handoff rather than pretending incompatible weights can be copied directly.

---

# 25. Interaction with working memory

Memory can help MacroCells become reusable functions rather than content memorization, but it introduces its own failure modes.

Canonical first memory design:

- small fixed number of typed/untyped slots;
- explicit read summary;
- explicit gated write;
- no attention over huge memory in V1.

Audit:

- write frequency;
- slot specialization;
- information retention across steps;
- ablate memory and measure capability loss;
- permute slots where semantics should be invariant;
- detect one-slot collapse.

A MacroCell should not need to encode all transient state in its weights.

---

# 26. Halting/adaptive steps

Adaptive execution is optional after the base cell/routing system works.

Do not add halting to the first MacroCell falsification run; it creates another credit-assignment variable.

Later:

`p_stop,t = sigmoid(H(x_t,m_t))`.

Measure accuracy vs average steps and enforce a compute penalty only after correctness is stable.

---

# 27. Error analysis requirements

Every failed run must classify errors where possible:

- wrong coarse address;
- correct family, wrong local cell;
- correct cell but insufficient body capacity;
- descriptor/body mismatch;
- memory read error;
- memory write corruption;
- composition/order error;
- early/late halt;
- OOD representation failure;
- kernel/runtime overhead only.

Do not summarize all failures as “router problem”.

---

# 28. Reproducibility requirements

Every serious run records:

```text
commit SHA
branch
teacher checkpoint
student checkpoint
architecture config
cell count
cell hidden size
candidate size
active k
address-tree geometry
physical params
active params
virtual params if any
seed
training tokens/steps
optimizer/schedule
loss weights
GPU model
GPU-hours
peak VRAM
system RAM
route entropy/utilization
quality metrics
OOD metrics
latency/throughput
failure classification
```

Negative results remain in `results/` and are not overwritten.

---

# 29. Anti-botqoqqa rules

This proposal is specifically designed to prevent endless combinations.

1. **One canonical MacroCell V1 first.** No attention/GRU/inner-router variants before the base gate.
2. **One routing hierarchy first.** Do not test dozens of trees before measuring the basic semantic-address hypothesis.
3. **Predeclare gates.** Failed gate blocks scale-up.
4. **Ablate one variable at a time.** Cell body, descriptor mechanism, routing, memory, and factorization are separate hypotheses.
5. **Oracle before router tuning.** If an oracle still requires many active cells, stop improving the router; representation is the bottleneck.
6. **Matched controls.** Always include compact dense and contiguous-chunk baselines where relevant.
7. **Equal compute.** Growth/continued-training claims require matched cumulative budget.
8. **Harder benchmark before larger bank.** Do not scale a saturated task.
9. **Real hardware before speed claim.** Active params are not latency.
10. **No novelty claim without prior-art review.** A positive engineering result is not automatically a new scientific principle.

---

# 30. Decision tree

```text
MacroCell local teacher imitation works?
  NO -> body design insufficient; STOP
  YES
   |
MacroCell at matched active compute beats contiguous chunks?
  NO -> sparse representation hypothesis weak; STOP/return to taklif15
  YES
   |
Beats compact dense control or gives clear systems advantage?
  NO -> sparse-cell complexity not justified
  YES
   |
Semantic descriptor/body consistency emerges?
  NO -> self-describing addressing rejected/rework descriptor mechanism only
  YES
   |
25–50% active budget preserves useful quality?
  NO -> central sparse-quality target not met
  YES
   |
Grouped kernel gives real latency/memory win?
  NO -> research architecture may remain scientifically interesting, no runtime claim
  YES
   |
Scale bank + capability frontier + OOD + multi-seed
```

---

# 31. What success would actually mean

A strong result would not be “we built a bigger neuron”. It would demonstrate all of the following together:

1. strong cell bodies can absorb distributed local computation;
2. cells organize into behaviorally meaningful reusable regions;
3. a cheap addresser retrieves a tiny candidate set without scanning the bank;
4. only a small active subset is needed for quality;
5. stored functional capacity can grow without proportional active compute;
6. growth moves a harder capability frontier, not only a saturated benchmark;
7. the sparse path is actually efficient on hardware.

Only this combination would support the project’s deeper hypothesis:

> Large stored knowledge can be organized as a sparse, semantically addressable library of reusable computational functions rather than as dense distributed computation that must be largely re-evaluated for every input.

Until those gates pass, MacroCells remain a promising but unproven research direction.
