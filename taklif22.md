# taklif22.md — Latent Computational Machine: separate knowledge, computation, working memory, and semantic control

Status: **canonical architecture thesis and staged falsification program; high priority; not yet validated**

> **IMPORTANT SCOPE RULE:** This proposal is architecture-level. It may use the independently tested MacroCell idea from `taklif19.md` / `taklif20.md`, but it must **not** simultaneously introduce the operator-valued parameterization from `taklif21.md`. `taklif21` remains isolated until it independently passes. Do not combine all proposals at once.

> **ANTI-BOTQOQ RULE:** The proposal is not permission to endlessly add modules. V1 has a fixed minimal graph, fixed measurements, mandatory controls, and explicit stop gates. Any extra attention block, GRU, inner router, extra memory type, additional loss, or new cell family requires a new ablation and must not be silently added to rescue a failed run.

---

# 0. Central thesis

The current Neural Engine evidence suggests three separate facts:

1. very small active computation can work on narrow structured problems;
2. routing a small subset of pretrained Qwen neuron/chunk fragments does not preserve teacher quality, even with a strong oracle, implying that those dense fragments are poor sparse primitives;
3. improving the router alone cannot solve a representation whose useful information is distributed across nearly all candidate fragments.

Therefore the next architecture should not start from the question

> “How can the router search a larger bag of tiny circuits?”

but from

> “What should be routable, what should be stored as knowledge, and what should remain in working state?”

The central hypothesis is:

`model capability ≈ reusable computation + factual/associative memory + working state + cheap semantic control`

rather than

`model capability ≈ enormous undifferentiated parameter field + increasingly clever router`.

The intended system is a learned sparse computational machine:

```text
input/context
    ↓
state encoder
    ↓
persistent working state
    ↓
need / latent-instruction generator
    ├──────────────→ long-term knowledge retrieval
    ↓                          ↓
semantic computation address ← retrieved context
    ↓
small candidate set
    ↓
Top-k computational cells
    ↓
READ → TRANSFORM → WRITE
    ↓
updated state
    ↓
repeat or halt
    ↓
output decoder
```

The model must learn a **latent computational language**: compact internal instructions that specify what kind of transformation is needed next, without requiring human-written labels such as ADD, SEARCH, COMPARE, or PARSE.

---

# 1. Four things that must not be conflated

The architecture explicitly separates four roles.

## 1.1 Computational skill

A computational cell implements a reusable transformation. Examples in human language may resemble retrieve, compare, bind, compose, transform, infer, update, but the final learned roles need not be human-interpretable.

Denote cell `i` by

`F_i : (s_t, W_t, c_t) -> (Δs_i, ΔW_i)`.

A cell should encode **how to transform information**, not memorize a large arbitrary table of unrelated facts.

## 1.2 Long-term knowledge

Long-term knowledge stores reusable content/associations that may change independently of the computational algorithm.

Abstract memory:

`K = {(k_j, v_j)}_{j=1..M}`

with keys `k_j ∈ R^p` and values `v_j ∈ R^v`.

Retrieval should return a small set of relevant values without scanning all `M` entries in the production design.

## 1.3 Working memory / state

Working memory stores intermediate values and bindings needed for the current computation:

`W_t ∈ R^(S × d_w)`.

This is not long-term factual storage. It is analogous to registers/scratch space.

## 1.4 Control / latent instruction

The controller decides what type of operation is needed next, not the answer itself.

`z_t = Q(s_t, pool(W_t), c_t)`

where `z_t ∈ R^p` is a latent computational instruction / need vector.

The hypothesis requires `Q` to remain small relative to the cell bank. If the controller itself becomes a giant dense model that secretly performs the task, the architecture has failed conceptually even if accuracy is high.

---

# 2. Minimal V1 state equations

Let:

- `x` = encoded input/context;
- `s_t ∈ R^d` = controller/local state;
- `W_t ∈ R^(S×d_w)` = working memory;
- `c_t ∈ R^d_c` = retrieved long-term-memory context;
- `z_t ∈ R^p` = latent computational instruction;
- `A_t` = active cell set, `|A_t| = k`;
- `T_max` = maximum adaptive computation steps.

Initialize:

`s_0 = E_x(x)`

`W_0 = E_w(x)`

`c_0 = 0`.

At step `t`:

`z_t = Q([s_t, pool(W_t), c_t])`.

The latent instruction is used for two potentially different addresses:

- computation address `z_t^C = P_C z_t`;
- knowledge-retrieval address `z_t^K = P_K z_t`.

This permits one internal intention to request both a computational transform and relevant content without forcing the same metric space to serve both purposes.

Long-term retrieval:

`c_t = Retrieve_K(z_t^K, K)`.

Cell candidate retrieval:

`C_t = Retrieve_C(z_t^C, {κ_i})`, with `|C_t| = L << N`.

Candidate self-match:

`a_i = g_i(z_t^C, s_t, pool(W_t), c_t)`, for `i ∈ C_t`.

Active set:

`A_t = TopK({a_i}, k)`.

For serial execution, preferred in the first real composition test:

`(s_t^(0), W_t^(0)) = (s_t, W_t)`

`(Δs_j, ΔW_j) = F_{i_j}(s_t^(j-1), W_t^(j-1), c_t)`

`s_t^(j) = s_t^(j-1) + γ_j Δs_j`

`W_t^(j) = W_t^(j-1) + WriteMask_j ⊙ ΔW_j`

for `j = 1..k`.

Then

`s_{t+1} = LN(s_t^(k))`

`W_{t+1} = Stabilize(W_t^(k))`.

Halt probability:

`h_t = σ(w_h^T [s_{t+1}, pool(W_{t+1})] + b_h)`.

The machine either halts or repeats until `T_max`.

Output:

`ŷ = D(s_T, pool(W_T), c_T)`.

---

# 3. Why knowledge and computation should be separated

Consider a function that answers a factual relation:

`Answer(entity, relation) = Decode(Retrieve(K, entity, relation))`.

If the fact changes while the algorithm remains the same, a clean architecture should update `K` without retraining every computational cell.

Conversely, if the algorithm changes while the facts remain fixed, one should be able to modify the computation path without rewriting the entire long-term memory.

This motivates a conditional-independence target:

`computation skill ⟂ arbitrary fact identity | required operation`

and

`memory content ⟂ execution policy | retrieval key`.

This will never be perfectly true in a neural model, but it can be measured experimentally.

## Required intervention test A — fact swap

Train on a synthetic or controlled knowledge table `K_A` with fixed computational tasks. Replace it with `K_B` while freezing computational cells and controller where possible.

Success criterion:

- answers that depend on changed facts should change accordingly;
- operation accuracy should remain stable;
- little or no cell retraining should be required.

If the system cannot use a new memory without retraining the cell bank, “knowledge/computation separation” is mostly cosmetic.

## Required intervention test B — operation swap

Keep facts fixed but introduce a new compositional rule/task. Adapt computation/controller while freezing long-term memory.

If memory has to be relearned for every computational change, the separation is weak.

---

# 4. Latent computational language

The architecture should not require hand-authored operator names. Instead, useful operation classes should emerge in a latent space.

Let `z_t = Q(...)`.

We want several invariances.

## 4.1 Content invariance

For two examples requiring the same abstract computation but different operands/content,

`sim(z_t(x_a), z_t(x_b))` should be high.

## 4.2 Functional separation

For examples requiring different transformations,

`sim(z_t(x_a), z_t(x_c))` should be lower when their computational role differs.

## 4.3 Compositional reuse

If a learned operation is useful in many programs, its latent region and corresponding cells should be reused across those programs.

Define a route-reuse metric:

`Reuse(role) = E[Jaccard(A_t^a, A_t^b) | same latent/known functional role, different content]`.

For synthetic tasks where the true operation is known only for evaluation, this can be measured without giving the label to the model.

A good latent language should show:

`Reuse_same_role >> Reuse_different_role`.

## 4.4 Avoid token-address memorization

Measure mutual information proxies between routes and raw value/token identity versus routes and functional role.

Desirable:

`I(Route; Role) >> I(Route; RawOperand)`

on datasets where operands vary independently of operation.

If route identity mostly tracks token/value identity, the system is memorizing content addresses, not learning a computational language.

---

# 5. Long-term knowledge memory V1

Do not begin with a billion-entry external memory. V1 should use a controlled differentiable key-value memory so failure can be diagnosed.

Memory entries:

`K = {(k_j, v_j)}_{j=1..M}`.

Dense oracle retrieval for small experiments:

`p_j = softmax((z^K · k_j)/τ)`

`c = Σ_j p_j v_j`.

This dense version is an **oracle/control**, not the production design.

Production-oriented retrieval must be hierarchical or approximate:

```text
z^K
 ↓
coarse code
 ↓
small bucket
 ↓
Top-r keys
 ↓
weighted retrieved context
```

Cost target:

`C_K(M) = o(M)`

and preferably near-logarithmic / near-constant over tested scaling ranges.

Report retrieval recall@r against the dense oracle so speed improvements are not confused with retrieval failure.

---

# 6. Computational cell interface

`taklif22` does not redefine MacroCell internals. It defines the interface that any accepted cell implementation must satisfy.

A cell is:

`F_i(s, W, c) -> (Δs, ΔW)`.

Mandatory properties:

1. read current state;
2. read a bounded summary/selection of working memory;
3. optionally consume retrieved long-term context `c`;
4. emit bounded residual state update;
5. emit controlled write proposal;
6. expose cheap descriptor/signature `κ_i` for semantic retrieval;
7. not contain an unrestricted global search over all cells/memory.

For the first architecture experiment, if `taklif20` is already independently GO, use its canonical mesoscopic MacroCell. If `taklif20` is not yet validated, use a simpler fixed-size cell and treat `taklif22` as a control-plane/memory-separation experiment.

Do not silently use `taklif21` operator-valued parameters here.

---

# 7. Semantic cell addressing

Full scan over `N` cell signatures:

`score_i = z^C · κ_i`

for all `i=1..N` costs `O(Np)` and is not accepted as the scalable architecture.

Use hierarchical addressing with branching factors `B_1,...,B_H` and leaf candidate size `L`.

Approximate route cost:

`C_route ≈ p Σ_h B_h + pL`.

If the tree is balanced and `N ≈ L Π_h B_h`, then the number of evaluated signatures can be dramatically smaller than `N`.

The architecture should separately report:

- exact dense-route oracle quality;
- hierarchical-route quality;
- retrieval recall of oracle Top-k;
- routing FLOPs;
- routing latency;
- route bytes transferred.

If dense routing works but hierarchical routing cannot recover the same useful cells, the architecture is not yet scalable.

---

# 8. Descriptor/body consistency

A cell address must correspond to the function the cell actually performs.

A free signature vector `κ_i` can drift independently from body `F_i`.

Use at least one of the consistency mechanisms from `taklif19`, preferably starting with a usage-derived prototype:

`p_i <- (1-β)p_i + β E[z_t^C | i selected and useful]`

and regularize

`L_desc = ||normalize(κ_i) - stopgrad(normalize(p_i))||^2`.

For a stronger audit, compute behavioral probes `P={u_1,...,u_J}`:

`b_i = H(F_i(u_1),...,F_i(u_J))`

and compare descriptor similarity to behavior similarity.

Desired correlation:

`corr(sim(κ_i,κ_j), sim(b_i,b_j)) > 0`.

A high routing score with no relation between descriptor geometry and behavior is a warning that the semantic-address story is false.

---

# 9. Working memory discipline

Working memory is a common source of hidden cheating. If `W_t` is huge, the controller can simply store the entire problem and solve it with a dense network.

Therefore V1 must cap:

- number of slots `S`;
- slot width `d_w`;
- write bandwidth per step;
- number of readable slots per selected cell where applicable.

Define write budget:

`B_write = number_of_modified_slots × d_w`.

Report quality as a function of `B_write`.

Required ablations:

- no persistent working memory;
- small memory;
- canonical memory;
- oversized memory control.

If all gains appear only with oversized memory, the claimed cell/computation architecture may not be doing the work.

---

# 10. Adaptive depth and planning

One hypothesis is that stronger reusable operations reduce effective routing/planning depth.

Let `T(x)` be number of outer computational steps and `k` active cells per step.

Effective routed decisions:

`D_route(x) = k T(x)`.

Track:

- mean `E[D_route]`;
- 95th percentile;
- distribution by task difficulty;
- accuracy conditioned on route depth.

A useful machine should increase depth on harder tasks and remain shallow on easy ones.

However adaptive depth can become a hidden compute explosion. Therefore report total executed cell MACs, not only active parameters per step.

---

# 11. Training problem: why naive joint training may collapse

All components are initially meaningless:

- controller does not know what operation is needed;
- cell signatures do not mean anything;
- cells have not specialized;
- memory keys are unstable;
- working-memory conventions are not established.

If everything is hard-routed from step 1, random early choices determine who receives gradient, producing dead cells and fragmented specialization.

Therefore V1 training must be staged.

---

# 12. Canonical training curriculum

## Phase A — state and memory plumbing

Use tasks with explicit intermediate-state targets or easily verified synthetic programs.

Train encoder, working-memory read/write, and decoder with broad/dense compute participation.

Goal: verify that the state machine can represent multi-step computation before adding sparse specialization.

## Phase B — soft computational mixture

Allow a larger candidate set with soft weights:

`y = Σ_{i∈C_t} p_i F_i(...)`.

Do not start with Top-1/Top-2.

Goal: give many cells useful gradient and allow primitive specialization to emerge.

## Phase C — latent-instruction alignment

Add route consistency / contrastive objectives so similar functional needs route to similar regions.

One possible loss for known synthetic role labels used **only during this diagnostic stage**:

`L_role = -log exp(sim(z,z+)/τ) / [exp(sim(z,z+)/τ)+Σ_- exp(sim(z,z-)/τ)]`.

For unsupervised settings, use successful-cell usage prototypes instead of labels.

The final architecture must not require human operation labels.

## Phase D — sparse annealing

Reduce active participation gradually, for example:

`Top-16 -> Top-8 -> Top-4 -> Top-2`.

Each transition requires a quality gate.

Do not continue to smaller `k` if the oracle itself fails at that budget.

## Phase E — knowledge separation

Introduce long-term retrieval and fact-swap interventions while freezing or partially freezing cell bodies.

Goal: test whether factual content truly lives in memory rather than being redundantly copied into cells.

## Phase F — end-to-end teacher distillation

Only after the controlled system works, use a pretrained Transformer/Qwen teacher.

Possible losses:

`L_hidden = ||P_s s_student - h_teacher||^2`

`L_KL = KL(p_teacher || p_student)`

`L_LM = CE(y_student, target)`.

Use progressively broader replacement:

late FFN/function -> late block -> several blocks -> full-stack student.

Do not start by replacing the entire Transformer.

---

# 13. Total training objective

A generic objective may be:

`L = λ_task L_task + λ_KL L_KL + λ_state L_state + λ_route L_route + λ_desc L_desc + λ_sparse L_sparse + λ_load L_load + λ_halt L_halt + λ_mem L_mem`.

But **do not enable all terms on day one**.

Each added term must have an ablation showing that it fixes a measured failure.

Potential definitions:

## Task / language loss

`L_task = CE(ŷ, y)`.

## Teacher KL

`L_KL = KL(softmax(t/τ_T) || softmax(s/τ_T))`.

## Sparsity / budget penalty

`L_sparse = E[max(0, C_active(x)-B_target)]`.

## Load-collapse penalty

Let `u_i` be normalized cell usage. Penalize only extreme collapse:

`L_load = Σ_i ReLU(u_i-u_max)^2 + ReLU(u_min-u_i)^2`

with permissive bounds, not forced uniform usage.

## Halt cost

`L_halt = η E[T(x)]`

subject to a quality constraint, to discourage endless computation.

## Memory-retrieval supervision/contrast

When a ground-truth memory item is known in controlled experiments:

`L_mem = -log p(j* | z^K)`.

This term should be removed or replaced with teacher/self-supervision in open-domain training.

---

# 14. Critical identifiability problem

The decomposition

`knowledge + computation + control`

is not uniquely identifiable.

A powerful cell can memorize facts. A powerful memory value can encode an entire program. A powerful controller can solve the task directly.

Therefore architectural claims require **interventions**, not only end accuracy.

Mandatory interventions:

1. freeze cells and swap facts;
2. freeze memory and change operation distribution;
3. scramble cell IDs but preserve signatures;
4. scramble signatures but preserve cell bodies;
5. zero long-term memory;
6. zero working memory;
7. replace controller with oracle route on synthetic tasks;
8. replace learned cells with random/frozen cells;
9. allow dense all-cell oracle;
10. compare against a compact dense model with matched active compute.

Without these controls, high accuracy does not prove the proposed decomposition.

---

# 15. Main failure modes

## F1 — controller becomes the real model

If `Q` is too large, it may perform the computation and use cells as decorative residuals.

Audit: controller-only ablation.

If controller-only retains most quality, shrink/freeze it or reject the architecture claim.

## F2 — cells become fact stores

Large cells may memorize arbitrary entities/facts.

Audit: fact-swap test and probe cell activation by entity identity.

If swapping memory does not update answers without cell retraining, separation failed.

## F3 — memory becomes a giant program store

Memory values may encode not just facts but computation traces/programs.

Audit: reuse the same memory facts under novel operations; measure whether cells/general control still matter.

## F4 — route fragmentation by surface content

Different values/tokens requiring the same operation may route to unrelated cells.

Audit: role-conditioned route reuse and operand-conditioned route entropy.

## F5 — all cells become generalists

If every cell learns similar behavior, semantic routing is unnecessary.

Audit: behavioral probe diversity and replace-one-cell tests.

## F6 — one/few cells monopolize all work

This collapses to a small dense model.

Audit usage histogram, entropy, and quality after removing dominant cells.

## F7 — descriptors lie

Signature similarity may not reflect functional similarity.

Audit descriptor/behavior correlation and signature-swap interventions.

## F8 — knowledge retrieval requires dense scan

Accuracy may look good with dense memory attention but fail scalability.

Audit dense-oracle vs hierarchical retrieval recall/latency.

## F9 — sparse control costs more than saved compute

Top-k may reduce cell FLOPs while increasing gathers, kernel launches, and memory traffic.

Audit real wall-clock and profiler counters. Active parameter count is not a speed claim.

## F10 — adaptive depth explodes

Hard tasks may require so many steps that total compute exceeds a Transformer.

Audit total MAC/token and latency/token, not per-step cost.

## F11 — catastrophic semantic drift

As cells change, signatures and controller queries chase one another, destabilizing addresses.

Mitigation: EMA prototypes, slower signature updates, staged freezing, periodic descriptor recomputation.

## F12 — specialization prevents transfer

Cells may become so narrow that unseen compositions cannot reuse them.

Audit novel program combinations and OOD operand/content shifts.

## F13 — apparent OOD gain is leakage

Procedural generators may share templates or token encodings across train/test.

Use structurally distinct generators, held-out operator combinations, held-out program graphs, and fresh seeds.

## F14 — teacher distillation merely copies local activations

A student may imitate a small teacher dataset but fail open-domain behavior.

Use independent corpora/benchmarks and teacher agreement on unseen text.

## F15 — architecture degenerates into ordinary MoE

If cells are giant layer-local FFNs, no persistent state is used, and routing occurs independently per Transformer layer, the result is essentially MoE.

This is not invalid engineering, but it is not evidence for the Latent Computational Machine thesis.

---

# 16. Required baseline families

Every serious result must include matched controls.

1. current micro-circuit NE;
2. accepted MacroCell model if `taklif20` is GO;
3. compact dense MLP/recurrent student with similar active scalar DOF;
4. small Transformer matched as reasonably as possible for train compute/parameter budget;
5. conventional MoE with comparable active compute where feasible;
6. random routing control;
7. dense-routing oracle;
8. oracle program/route on synthetic tasks where ground truth is known;
9. no-long-term-memory model;
10. no-working-memory model.

Do not compare only against an intentionally weak baseline.

---

# 17. Evaluation ladder

The architecture must progress through increasingly difficult tests.

## Level 1 — controlled primitive reuse

Tasks with known reusable operations and independently varied operands.

Question: do routes depend on operation more than raw value?

## Level 2 — multi-step programs

4-step, 6-step, 8-step and variable-depth programs.

Question: does adaptive routing compose learned primitives?

## Level 3 — novel program combinations

Train on a subset of valid operation graphs; test unseen compositions built from known primitives.

Question: are cells reusable or memorized program templates?

## Level 4 — knowledge/computation interventions

Swap fact tables, add new facts, change relation values, preserve operations.

Question: does factual knowledge live in the intended subsystem?

## Level 5 — representation OOD

Change value ranges, encodings, symbol identities, and graph structures.

Question: has the earlier `0–31 -> 32–63` type failure improved?

## Level 6 — Qwen local function distillation

Replace late FFN/block functions with the new machine while preserving teacher behavior.

Question: can dense distributed language computation be re-expressed as sparse reusable operations?

## Level 7 — broader language

Use held-out natural text, perplexity/CE, top-1 teacher agreement, KL, downstream tasks, and real latency.

Do not scale to larger Qwen models before 0.6B-scale reproducibility.

---

# 18. Main scientific metrics

Accuracy alone is insufficient.

Report:

- task accuracy / CE / perplexity;
- teacher KL and top-1 agreement;
- active cell count;
- active scalar parameters;
- total executed MACs/token;
- route cost/token;
- memory retrieval cost/token;
- total latency/token;
- peak VRAM/RAM;
- stored physical parameters;
- virtual capacity if any, clearly labeled;
- route reuse by functional role;
- route sensitivity to raw operands/tokens;
- cell usage entropy;
- dead-cell fraction;
- descriptor/behavior correlation;
- long-term-memory retrieval recall;
- working-memory write/read bandwidth;
- average and tail computation depth;
- OOD and intervention results.

Never compress these into one headline number.

---

# 19. Formal success criterion

Let

`Q_model(B_active, C_total, L_latency, G_ood)`

represent quality under active parameter budget, total executed compute, latency, and OOD/generalization constraints.

The proposal is useful only if it improves the Pareto frontier, not merely one metric.

A strong result should satisfy some version of:

`Quality_LCM >= Quality_control - ε`

while

`ActiveCompute_LCM << ActiveCompute_dense`

and/or

`OOD_LCM > OOD_micro`

and

`RouteCost_LCM = sublinear(N_cells)`

and

`MemoryCost_LCM = sublinear(M_entries)`

with real wall-clock evidence once kernels are optimized.

No single inequality is sufficient by itself.

---

# 20. Canonical V1 experiment

Do **not** begin with open-domain language.

Use a synthetic learned-machine benchmark with:

- independent factual table;
- 8–16 reusable primitive transformations;
- variable 2–8 step programs;
- held-out program compositions;
- held-out fact swaps;
- held-out symbol/value encodings;
- enough entropy to prevent template memorization.

Suggested architecture scale for first diagnostic run:

- state `d = 384`;
- working memory `S = 8` slots, `d_w = 384`;
- latent instruction `p = 128`;
- 32 or 64 computational cells initially;
- soft routing during warm-up;
- final `Top-2` or the smallest `k` that passes the oracle gate;
- memory table small enough to permit dense retrieval oracle plus hierarchical candidate retrieval control;
- adaptive outer depth capped at 8 or 12 steps.

This first run is not a scale claim. It tests whether the decomposition itself emerges.

---

# 21. Mandatory oracle gates before expensive training

## Oracle A — sparse cell upper bound

Given trained cell bodies, allow an expensive oracle to choose the best subset for each example.

If even oracle Top-2/Top-4 cannot preserve quality, do not tune the router endlessly. The cell representation is insufficient at that sparsity.

## Oracle B — memory retrieval upper bound

Use dense exact search over all memory entries.

If dense retrieval fails, hierarchical retrieval is not the problem.

## Oracle C — program route

On synthetic tasks with known program steps, feed the correct operation route while using learned cells.

If quality is poor, cell/state representation is the bottleneck, not control.

## Oracle D — perfect facts

Provide the correct fact value directly while leaving computation learned.

If performance remains poor, long-term memory retrieval is not the main issue.

These gates localize failures before architecture search begins.

---

# 22. GO / NO-GO gates

## GO-1 — computational reuse

Proceed only if same functional roles across different operands route to substantially more similar cells/addresses than different roles.

## GO-2 — sparse oracle

Proceed to router optimization only if oracle sparse selection is close to dense-cell quality at the intended active budget.

## GO-3 — fact modularity

Proceed only if a substantial portion of fact changes can be incorporated by memory replacement/update without retraining computational cells.

## GO-4 — composition

Proceed only if unseen compositions of known primitives materially outperform memorization/random-route controls.

## GO-5 — OOD

Proceed to language transplant only if representation OOD is better than the current micro-cell baseline on genuinely held-out encodings/ranges/graphs.

## GO-6 — scalable addressing

Dense semantic routing may be used for diagnosis, but scale claims require hierarchical/sublinear route and memory retrieval.

## GO-7 — real compute

Do not claim efficiency from active parameter counts. A speed claim requires measured wall-clock benefit with grouped/fused kernels or otherwise realistic implementation.

## NO-GO

Freeze/reject this architecture direction if, after controlled training and reasonable tuning:

- cells fail to specialize/reuse;
- oracle sparse selection still needs most cells;
- controller-only retains most capability;
- fact swap requires widespread cell retraining;
- unseen compositions do not improve over dense compact control;
- addressing/retrieval cost grows approximately linearly with bank size;
- total adaptive compute exceeds dense baselines with no quality/generalization advantage.

Do **not** rescue a NO-GO by adding arbitrary modules without a new proposal.

---

# 23. Transformer/Qwen -> LCM functional transplant

If controlled tests pass, the intended transfer is **function-level**, not weight-by-weight.

Teacher block/function:

`T_l(h)`.

Student machine segment:

`S_l(h, W; θ)`.

Train:

`min_θ E ||P S_l(h,W)-T_l(h)||^2 + λ_KL KL(p_T || p_S)`.

Progressive scaffold removal:

```text
100% teacher / 0% student
        ↓
teacher + student residual
        ↓
teacher frozen, student imitates
        ↓
gradual student contribution increase
        ↓
student-only block/function
```

Critical measurement: whether the student learns reusable latent operations across many teacher contexts, rather than one hidden imitation network per layer.

A local successful transplant is not proof of full language-model replacement.

---

# 24. Why this may work

The architecture has several plausible advantages.

## 24.1 Better routing granularity

A routing decision selects a meaningful learned transformation instead of a tiny fragment. This may reduce planning depth and credit fragmentation.

## 24.2 Reuse across content

If factual content lives mainly in memory, the same computational cell can operate on many entities/values.

## 24.3 Better modular interventions

Knowledge can potentially be updated without full retraining, and computational skills can be improved without rewriting all facts.

## 24.4 Sparse execution designed from training start

Unlike pruning Qwen neuron chunks after dense pretraining, the entire representation is trained to be sparse and semantically addressable from the beginning.

## 24.5 Potential route scalability

A structured semantic address may replace full-bank expert scoring.

These are hypotheses, not established benefits.

---

# 25. Why this may fail fundamentally

The most serious possibility is that natural-language intelligence is intrinsically highly distributed: useful transformations may not decompose into a small library of reusable latent operations at the desired fidelity.

If so:

- sparse oracle will need many cells;
- latent instruction space will fragment by content;
- fact/computation separation will be blurry;
- adaptive depth may explode;
- compact dense models may remain superior.

Another possibility is that the desired decomposition exists but SGD cannot discover it reliably without stronger inductive bias or supervision.

A third possibility is systems-level: the mathematical architecture may reduce theoretical compute but perform poorly on GPUs because dynamic routing/memory access destroys arithmetic intensity.

These outcomes must be treated as valid scientific results, not reasons to keep changing the architecture indefinitely.

---

# 26. Relationship to previous proposals

- `taklif15`: learned latent reusable basis — complementary representation idea.
- `taklif16`: Transformer -> NE cross-architecture transplant — becomes the transfer mechanism after this architecture passes controlled tests.
- `taklif17`: capability-frontier/OOD protocol — mandatory evaluation framework here.
- `taklif18`: hardware-native grouped sparse runtime — required for eventual speed claims.
- `taklif19`: self-describing semantic MacroCells — compatible cell/address mechanism.
- `taklif20`: mesoscopic MacroCell granularity — candidate computational primitive, but must retain its own independent evidence.
- `taklif21`: operator-valued parameters — **must remain isolated at first**; do not mix into this architecture until independently GO.

`taklif22` is the architecture-level synthesis, but it does not override the independent falsification requirements of the earlier proposals.

---

# 27. Agent execution order

For an autonomous research agent:

1. verify prior proposal results and stop gates;
2. build the controlled `taklif22` benchmark first;
3. implement minimal state/working-memory machine;
4. establish dense/oracle routes and memory retrieval;
5. test operation-vs-content route reuse;
6. run fact-swap and operation-swap interventions;
7. add soft cell specialization;
8. run sparse oracle before router optimization;
9. anneal to hard sparse routing only if oracle passes;
10. test hierarchical addressing and memory retrieval;
11. test 4/6/8-step and unseen program compositions;
12. test representation OOD;
13. only after all relevant gates pass, attempt Qwen local functional transplant;
14. do not add `taklif21` parameter algebra in these runs;
15. log negative results exactly, do not silently retune the benchmark.

---

# 28. Final research question

The proposal reduces the project to a precise falsifiable question:

> Can a neural system learn a compact latent instruction space and a reusable sparse library of computational transformations, while storing mutable factual/associative content separately enough that only a very small amount of computation needs to activate for each step?

In shorthand:

`dense distributed intelligence`

`-> reusable latent operations + memory + sparse control`.

If the answer is yes under controlled OOD, intervention, and real-compute tests, this would be a materially different and more scalable architecture direction than merely routing smaller pieces of a dense Transformer.

If the answer is no even for strong sparse oracles and controlled tasks, the project should stop investing in ever more elaborate routers and revisit the core sparse-decomposition hypothesis.