# taklif23.md — Foundational Computational Substrate: redefine the learning object, execution graph, memory, and scaling law

Status: **canonical high-risk/high-reward foundational research program; staged falsification only; independent from the current Neural Engine production path**

> **SCOPE:** This proposal is intentionally more fundamental than a new router, MacroCell, sparse FFN, or Transformer replacement block. It asks whether the default mathematical object of AI should remain a fixed globally differentiable network of scalar parameters at all. It does **not** ban gradients, matrices, neural modules, attention, or backpropagation. Those may remain useful local tools. What changes is that none of them are assumed to be the global organizing principle.

> **ANTI-BOTQOQ RULE:** This document is a falsification program, not permission to endlessly invent modules. Every stage has a fixed hypothesis, fixed controls, fixed measurements, a small rescue budget, and an explicit stop decision. A failed foundation must not be kept alive by silently adding attention, recurrence, auxiliary losses, larger hidden widths, more memory, more training, or a stronger controller.

> **SEPARATION RULE:** Do not merge this proposal with `taklif19.md`, `taklif20.md`, `taklif21.md`, or `taklif22.md` in the first gates. Those proposals may later provide components only after this foundation independently demonstrates an advantage. `taklif23` must first prove that its **learning organization** is useful, not borrow enough machinery from existing systems to hide failure.

> **HARDWARE RULE:** No new hardware is required or allowed as an explanation for an early failure. V0 must run on ordinary CPU/GPU/RAM. Custom hardware may be considered only after a software implementation proves a repeatable algorithmic advantage.

---

# 0. Why this proposal exists

The project has repeatedly found that many architecture-level ideas can work locally yet fail at a more global boundary:

- sparse capacity can help while active compute stays nearly fixed, but can plateau;
- large expressive cells can become difficult to train or route;
- structured/operator-valued parameters can represent structured targets extremely efficiently, yet a particular placement can damage a real model;
- pretrained dense functions can be transferred exactly, yet naive sparsification of their internal neuron fragments can fail on genuinely held-out language;
- routing may be correct while the object being routed is the wrong primitive;
- memory, computation, and working state become easier to reason about when separated.

These failures suggest a deeper possibility: the bottleneck may not be only the choice of layer or router. The project may still be optimizing inside an inherited definition of a neural model:

`model = fixed graph of differentiable tensor operations with a large scalar parameter field`

and

`learning = minimize one global loss by propagating credit through that graph`.

`taklif23` asks a more fundamental question:

> **What if scalar parameters, fixed layer graphs, monolithic knowledge storage, and globally uniform credit assignment are implementation choices rather than the correct primitive definition of an intelligent machine?**

The target is not “non-Transformer for the sake of being non-Transformer.” The target is a system whose natural scaling unit is a **reusable computational structure** rather than merely another scalar weight.

---

# 1. The old assumptions under test

The proposal does not claim these assumptions are wrong. It explicitly tests whether relaxing them creates a measurable advantage.

## A1 — Parameter primacy

Conventional deep learning treats a scalar `w` as the atomic learned object.

A larger function is represented by many such scalars:

`θ = (w_1, ..., w_N)`.

`taklif23` instead treats scalar parameters as an implementation detail inside a richer learned object.

## A2 — Fixed graph primacy

Conventional networks normally train a graph whose large-scale topology is designed before training.

`taklif23` permits the execution graph itself to be selected, composed, compressed, split, reused, and eventually grown during learning.

## A3 — Global differentiability primacy

Backpropagation is an extremely efficient credit-assignment method when the whole useful computation is differentiable and fixed.

`taklif23` does not remove it. Instead it asks whether **local parameter optimization** and **structural credit assignment** should be separate mechanisms.

## A4 — Knowledge/computation co-location

Dense language models use the same parameter field for procedural computation, factual associations, style, syntax, and many other roles.

`taklif23` requires explicit tests of whether mutable factual/associative knowledge can be stored separately from reusable computation.

## A5 — Capacity implies active work

For a dense model, adding capacity usually also raises active computation and memory traffic.

The desired alternative is:

`stored capability ↑↑ while active execution cost stays nearly fixed or grows slowly`.

## A6 — One representation serves all roles

A single hidden vector is often asked to encode data, control, intermediate state, and retrieval keys.

`taklif23` permits typed state components with different semantics and update rules.

---

# 2. Central thesis

The proposed machine is an **Adaptive Computational Substrate (ACS)**.

Its core abstraction is:

`intelligence = typed state + external knowledge + reusable learned operators + dynamic execution graph + structural/local learning`

rather than:

`intelligence = one giant static differentiable parameter field`.

The substrate contains five first-class objects:

1. **typed working state** `S`;
2. **external/associative memory** `M`;
3. **operator library** `O = {O_i}`;
4. **execution policy / graph constructor** `Π`;
5. **learning system** that modifies both local operator parameters and the library/graph structure.

The first canonical equation is:

`(S_{t+1}, y_t) = O_{i_t}(S_t, R_t; θ_{i_t})`

where

`R_t = Retrieve(M, q_t)`

and

`i_t = Π(S_t, q_t, H_t)`.

`H_t` is execution history / structural context.

Unlike a fixed-depth neural stack, `i_t` may vary with the input and the system may halt when its state is sufficient.

---

# 3. The new atomic learned object

The atomic learned object is not required to be one scalar and is not required to be one neural layer.

Define operator `O_i` as:

`O_i = (σ_i, θ_i, κ_i, ρ_i, ω_i, c_i)`

where:

- `σ_i` = typed interface/signature;
- `θ_i` = local internal parameters, possibly neural, symbolic, linear, table-based, or mixed;
- `κ_i` = descriptor/address used to retrieve the operator;
- `ρ_i` = read policy over state/memory;
- `ω_i` = write policy;
- `c_i` = execution-cost metadata.

The signature is:

`σ_i : T_in -> T_out`

or for multi-input state:

`σ_i : (T_1 × ... × T_r) -> (T'_1 × ... × T'_w)`.

The first prototype should use a small finite type set, for example:

- scalar/value;
- vector/value;
- boolean/control;
- key/address;
- sequence/slot;
- relation/binding.

Types are not necessarily human semantic labels in the final system. V0 uses explicit types only to prevent meaningless compositions and to make failure diagnosable.

---

# 4. Typed state instead of one undifferentiated hidden vector

Let working state be a product space:

`S_t = V_t × B_t × K_t × C_t`

where:

- `V_t` = value registers;
- `B_t` = bindings/relations;
- `K_t` = temporary keys/retrieval references;
- `C_t` = control/continuation state.

A simpler V0 may use only two or three components.

The key rule is that an operator cannot arbitrarily rewrite all state. Its signature specifies legal reads/writes.

If `R_i` is the set of readable slots and `W_i` the writable slots:

`O_i : S_t[R_i] × R_t -> ΔS_t[W_i]`.

Update:

`S_{t+1}[W_i] = Stabilize(S_t[W_i] + ΔS_t[W_i])`.

All untouched state is copied exactly.

This creates a measurable notion of **write bandwidth**:

`B_write(i) = |W_i| × state_width`.

Every benchmark must report it. A system that “wins” only because every operator rewrites the entire state is not a sparse computational substrate.

---

# 5. External knowledge is a first-class object

Long-term mutable knowledge is represented as:

`M = {(k_j, v_j, meta_j)}_{j=1..N_M}`.

The minimal retrieval equation is:

`r_t = Retrieve(q_t, M)`.

For a small oracle:

`p_j = softmax(sim(q_t, k_j)/τ)`

`r_t = Σ_j p_j v_j`.

Production scaling must later use sublinear retrieval, but V0 must first separate correctness from indexing complexity.

The central intervention is:

`M_A -> M_B`

while freezing the computational library.

If answers change with the memory while the reusable algorithm remains stable, the system has genuine knowledge/computation separation.

If a fact swap requires broad operator retraining, the separation claim fails.

---

# 6. Execution is a temporary program, not a fixed layer stack

For input `x`, the machine constructs a sequence or DAG:

`G_x = (O_{i_1}, O_{i_2}, ..., O_{i_T})`

subject to type compatibility.

Serial V0:

`S_{t+1} = O_{i_t}(S_t, R_t)`.

Later DAG form:

`G_x = (V_x, E_x)`

where nodes are operator invocations and edges carry typed values/state references.

The machine halts with:

`h_t = Halt(S_t, H_t)`.

V0 may use a fixed maximum `T_max` with an explicit halt action.

A central metric is:

`D_exec(x) = number of operator invocations`

and total active cost:

`C_exec(x) = Σ_{t=1..T} c_{i_t}`.

The system is not allowed to hide increasing computation behind “only one active operator per step” while making `T` explode.

---

# 7. Learning is split into two levels

This is the most important mathematical departure.

## 7.1 Local parameter learning

If operator `O_i` contains differentiable parameters `θ_i`, optimize them with any appropriate method:

`θ_i <- θ_i - η ∇_{θ_i} L_local`.

Backpropagation is allowed inside a selected operator or a small differentiable subgraph.

No ideological ban on gradients exists.

## 7.2 Structural learning

The system separately learns:

- which operators exist;
- which are retrieved;
- which are composed;
- which sequences deserve compression into a macro-operator;
- which operators should split, merge, prune, or clone;
- which state interfaces are useful.

Structural decisions need not be differentiable.

Let a candidate structural edit be `e` applied to graph/library `G`:

`G' = e(G)`.

Define structural utility:

`U(e) = E[L(G) - L(G')] - λ_C ΔC_exec - λ_M ΔC_storage - λ_W ΔB_write + λ_R ΔReuse`.

Accept the edit when its held-out expected utility is reliably positive.

This turns architecture change into an explicit learned/searchable object instead of a human-only outer loop.

---

# 8. Structural credit assignment

The main risk of abandoning a purely fixed differentiable graph is combinatorial credit assignment.

Therefore V0 requires explicit structural credit estimates.

For an invoked operator `O_i`, approximate counterfactual contribution:

`Δ_i = L(G_without_i_or_replaced) - L(G)`.

Because exact Shapley-style attribution is too expensive, use randomized local interventions:

- replace `O_i` with identity/no-op if type-safe;
- replace it with another compatible operator;
- skip it;
- replay with its output detached/frozen;
- perturb only its selected state write.

Estimate:

`Credit_i = E[Δ_i | contexts where i was active]`.

For a transition `(i -> j)`:

`Credit_{i,j} = E[L(G_without_transition) - L(G)]`.

Maintain exponentially weighted statistics:

`C_i <- (1-β) C_i + β Δ_i`.

`C_{ij} <- (1-β) C_{ij} + β Δ_{ij}`.

This credit is not a replacement for all gradients. It governs **structural retention, composition, and search**.

---

# 9. The global objective is multi-axis, not only task loss

The canonical objective for comparing complete systems is:

`J = L_task + λ_E C_exec + λ_S C_storage + λ_W B_write + λ_D D_exec + λ_I C_index - λ_R R_reuse + λ_F F_edit`.

Definitions:

- `L_task` = held-out task loss;
- `C_exec` = active MAC/FLOP/byte cost;
- `C_storage` = physical stored bytes/DOF;
- `B_write` = state write bandwidth;
- `D_exec` = number of routed decisions;
- `C_index` = routing/retrieval cost;
- `R_reuse` = cross-task reuse score;
- `F_edit` = successful fact/skill editability score.

Do **not** train directly on a heavily tuned weighted sum in V0. The equation is first an accounting framework. Early training should minimize task losses with a few controlled regularizers. The full `J` is used to compare architectures and prevent hidden cost shifts.

---

# 10. New scaling law target

Let:

- `C_store` = total stored structural capacity / bytes / scalar-equivalent DOF;
- `C_active` = active execution cost per example/token;
- `P` = held-out quality;
- `R` = reuse/generalization score.

The primary scaling coefficient is:

`S_cap = ΔP / Δlog(C_store)`

subject to:

`|Δlog(C_active)| <= ε`.

The project-level target is:

`S_cap > 0`

across multiple consecutive capacity increases, not one cherry-picked jump.

A stronger score is:

`S_reuse = ΔR / Δlog(C_store)`.

A useful foundation should show a frontier like:

`32 operators -> 128 -> 512 -> 2048`

with active operators per step fixed or nearly fixed while capability/quality improves.

A single jump followed by saturation is a positive clue, not a scaling law.

---

# 11. Operator creation: the machine may grow its own vocabulary

The system begins with a small primitive library.

If a subsequence appears repeatedly and has positive structural credit:

`O_a -> O_b -> O_c`

propose a macro-operator:

`O_m ≈ O_c ∘ O_b ∘ O_a`.

Creation procedure:

1. collect successful traces containing the subsequence;
2. record subgraph input/output state pairs;
3. fit a candidate `O_m` to reproduce the subgraph transform;
4. evaluate on held-out traces from multiple task families;
5. replace the original sequence only when quality stays within tolerance and execution cost decreases;
6. retain the original primitives until the macro proves reusable.

Macro creation utility:

`U_macro = ΔL - λ_E ΔC_exec - λ_D ΔD_exec + λ_R ΔReuse`.

A successful macro should reduce routed depth without simply memorizing a narrow task.

---

# 12. Operator split, merge, clone, and prune

## Split

If one operator has high conditional error variance across contexts:

`Var(L_i | context) >> threshold`

and two context clusters have different optimal transforms, propose specialization:

`O_i -> {O_i^A, O_i^B}`.

## Merge

If two operators have highly similar behavior on a probe set and merging does not hurt held-out quality:

`sim_behavior(O_i, O_j) >= τ_merge`

propose a shared operator.

## Clone

If one operator is overloaded across incompatible roles but is strongly useful, clone before specialization rather than mutating the only copy.

## Prune

Prune when all are true over a sufficient evaluation window:

- low selection frequency;
- low positive counterfactual credit;
- no unique held-out task contribution;
- replacement by another operator causes negligible loss.

No operator is pruned only because its raw routing frequency is low.

---

# 13. Operator representation families allowed in V0/V1

`taklif23` is a foundation test, so the operator internals must stay deliberately simple at first.

Allowed V0 operator families:

1. small linear/residual map;
2. small 2-layer MLP;
3. gated low-rank map;
4. exact deterministic primitive for diagnostic controls.

Do not begin with giant MacroCells, attention inside every operator, recursive inner routing, or operator-valued parameters.

Only after the foundation passes may an operator use:

- `taklif21` operator-valued structure;
- learned macroparameters;
- more complex internal recurrence;
- compiled Qwen-derived transforms.

This isolates the value of the **substrate organization** from the expressivity of the operator body.

---

# 14. Controller / proposer must remain weak enough to audit

The controller proposes the next operation but must not secretly solve the whole task.

Canonical V0:

`q_t = Q(S_t, x_summary)`

`score_i = sim(q_t, κ_i) + type_mask(i)`.

`i_t = TopK(score, k=1)` for the first serial gate.

Later:

`k in {1,2,4}`.

Controller constraints:

- parameter count <= a fixed small fraction of the library;
- no direct access to the final answer target;
- no full unrestricted memory table scan in production gates;
- controller-only ablation must be weak.

Required ablation:

`operators = identity/no-op`.

If the controller still solves the task, the architecture has hidden the computation in `Q` and fails conceptually.

---

# 15. Routing is not assumed to be neural

Candidate selection may use:

- learned cosine descriptors;
- hierarchical tree indexes;
- locality-sensitive hashing;
- typed symbolic filters;
- bandit statistics;
- learned query encoder;
- combinations of the above.

The scalable requirement is:

`C_route(N) = o(N)`

for large operator bank size `N`.

Dense full scan is allowed only as an oracle in small experiments.

Report:

- oracle route quality;
- approximate route recall;
- route FLOPs;
- bytes read;
- wall-clock time;
- route stability under small numeric perturbations.

---

# 16. Three distinct memories

Do not use one giant tensor and call everything memory.

## 16.1 Factual/associative memory

Mutable external knowledge.

## 16.2 Working memory

Short-lived intermediate state for the current problem.

## 16.3 Structural memory

Statistics about successful operators/transitions/macros:

`C_i, C_{ij}, usage_i, contexts_i, failure_i`.

Structural memory is used to propose edits and retrieval priors, not to answer factual questions directly.

This separation must be audited with swaps and freezes.

---

# 17. Continual learning target

A major hypothesized advantage is that a new skill should be addable without rewriting all old skills.

Suppose library after tasks `1..n` is `O^(n)`.

Train task `n+1` while freezing most existing operators and allowing:

- new operators;
- small targeted edits to selected operators;
- controller/index updates.

Define backward retention:

`Retain = P_old_after / P_old_before`.

Define new-skill efficiency:

`E_new = ΔP_new / added_storage`.

The foundation is interesting only if it shows substantially lower catastrophic interference than a matched monolithic baseline under an equal adaptation budget.

---

# 18. Knowledge editing target

For a fact set `F`, replace a subset `F_edit` in memory without changing computational operators.

Measure:

- edited fact accuracy;
- unaffected fact accuracy;
- procedural/computation accuracy;
- number of trainable parameters touched;
- latency to apply the edit.

Desired qualitative behavior:

`fact update -> memory write`

not

`fact update -> broad model retraining`.

This is a core advantage target, not an optional feature.

---

# 19. Self-compilation / hierarchy target

If the machine repeatedly executes:

`O_3 -> O_8 -> O_8 -> O_17`

and the sequence generalizes as a reusable transform, the system should be able to compile it into `O_91`.

Then higher-level sequences may compile again.

The intended hierarchy is:

`primitive -> operator -> macro-operator -> reusable algorithm -> skill`

The hierarchy must earn its existence by reducing execution depth/cost or improving generalization.

Do not create hierarchy merely because frequent patterns exist.

A macro is accepted only if it passes cross-context held-out tests.

---

# 20. Candidate learning mechanisms: what is allowed and what is not assumed

The research program may compare several structural-learning mechanisms.

## Mechanism A — local gradient + discrete structural search (canonical first choice)

- gradients train `θ_i` locally;
- a bandit/search process proposes route/library edits;
- counterfactual replay assigns structural credit.

This is the primary V0 because it preserves the strongest known local optimizer while changing the global organization.

## Mechanism B — population/beam search over execution graphs

Maintain a small beam of candidate typed programs and distill recurring useful subgraphs.

Use only for small synthetic tasks because combinatorial cost can explode.

## Mechanism C — local predictive objectives

Operators predict their target state transition or local residual instead of receiving one monolithic global gradient.

This is an ablation after Mechanism A, not an excuse to redesign everything at once.

## Mechanism D — energy/constraint relaxation

Represent a solution as a low-energy consistent state and let local operators reduce constraint violations.

This is a separate future branch. Do not mix it into V0 unless A fails for a diagnosed reason specifically related to sequential credit.

---

# 21. Canonical V0 benchmark family

Do not start with natural language.

Create procedurally generated tasks with known hidden programs, varied operands, and controllable depth.

Task families should include at least:

1. arithmetic transforms;
2. boolean/logic transforms;
3. permutation/string transforms;
4. key-value retrieval plus computation;
5. relation/binding tasks;
6. mixed programs requiring multiple operator types.

Critical split:

- train program lengths: `2..4`;
- held-out lengths: `5..8`;
- held-out compositions: operator sequences never seen in training;
- held-out operands/entities;
- held-out memory contents.

Ground-truth operator labels may be logged **for evaluation only**. They must not be provided to the final routing system except in explicit oracle controls.

---

# 22. Mandatory baselines

Every major gate must include matched controls.

## B0 — dense MLP/recurrent baseline

Approximately matched active compute.

## B1 — small Transformer baseline

Matched training budget and comparable active FLOPs where feasible.

## B2 — ordinary top-k MoE

Matched stored/active parameter budget.

## B3 — fixed operator library with learned router

Tests whether structural growth is actually useful.

## B4 — oracle program/router

Tests whether operator bodies/state representation can solve the task when routing is perfect.

## B5 — random routing

Detects whether apparent route semantics are unnecessary.

## B6 — controller-only

Operators disabled or identity.

## B7 — no external memory

Tests whether memory is causally useful.

No claim is accepted without the relevant baseline.

---

# 23. Gate 0 — implementation sanity

Purpose: verify the substrate machinery before learning anything difficult.

Use deterministic operators with known functions and an oracle controller.

Must pass:

- exact type checking;
- deterministic replay;
- state writes affect only allowed slots;
- operator replacement counterfactuals work;
- memory swap changes only expected outputs;
- execution accounting matches measured operator calls;
- no hidden full-bank scan in the “sparse” path.

Decision:

**FAIL = implementation bug, not research failure. Fix only correctness. No architecture additions.**

---

# 24. Gate 1 — learnability

Question:

> Can small learned operators and a weak controller learn simple tasks at all?

Protocol:

- `N=32` operators;
- Top-1 serial execution;
- fixed max depth;
- no structural growth;
- ordinary small MLP operators;
- two seeds for smoke, then three seeds for gate.

Pass requirement:

- train task solved to near-ceiling;
- held-out operands stable;
- quality no worse than a matched small recurrent/MLP control by more than a small tolerance;
- controller-only ablation clearly fails;
- oracle routing materially improves or matches learned routing.

Hard reject signal:

If repeated simple tasks cannot be learned without a large dense controller or massive operator bodies, this foundation has no useful starting point.

Rescue budget: **maximum 3 mechanism-level changes**.

Allowed rescues:

1. state normalization/stabilization;
2. soft-to-hard routing curriculum;
3. one local operator-body family change.

After three failed rescues: archive V0 and revisit the mathematical premise, not hyperparameters.

---

# 25. Gate 2 — unseen compositional generalization

Question:

> Does the system reuse learned transforms rather than memorize training programs?

Train:

- program depth `2..4`;
- subset of legal compositions.

Test:

- depth `5..8`;
- new compositions built from known primitives;
- new operands/entities.

Required metrics:

- exact task accuracy / MSE;
- route/program edit distance from oracle hidden program for diagnostics only;
- same-function route reuse;
- different-function route overlap;
- depth generalization curve;
- active cost by depth.

Pass signal:

- clear held-out-composition advantage over matched dense and ordinary MoE controls;
- no catastrophic collapse immediately beyond train depth;
- route reuse correlates with functional role more than raw operand identity.

If only in-distribution programs work, do **not** scale operator count.

---

# 26. Gate 3 — fixed-active capacity scaling

This is the central gate.

Use a fixed task family whose baseline is not saturated.

Scale stored library:

`N = 32 -> 128 -> 512 -> 2048`

while holding approximately fixed:

- active operators per step;
- operator body size;
- maximum/mean execution depth budget;
- controller size or controller FLOPs;
- training data distribution.

Report:

`P(N), C_store(N), C_active(N), route_cost(N)`.

Compute:

`S_cap = ΔP / Δlog(C_store)`.

Minimum evidence for “alive”:

- positive quality/capability movement across at least two consecutive scale jumps;
- no corresponding near-linear active-cost growth;
- effect repeats across at least three seeds at the strongest two scale points;
- route/index cost remains sublinear enough that the gain is not erased.

Strong evidence:

`P_32 < P_128 < P_512 < P_2048`

with `C_active` within roughly ±10–20% depending on unavoidable index overhead.

A single `32 -> 128` improvement followed by flat results is recorded as a bounded useful regime, not a scaling breakthrough.

---

# 27. Gate 4 — memory/computation separation

Train computation with memory table `M_A`.

Evaluate:

1. original `M_A`;
2. fully replaced fact values `M_B`;
3. held-out keys/entities where the addressing scheme permits;
4. zero-memory ablation.

Pass requires:

- changed facts produce changed correct answers;
- computational skill remains stable;
- zero-memory degrades memory-dependent tasks;
- memory edits do not require broad operator retraining;
- held-out addressing is tested so per-entity memorization cannot masquerade as general memory.

Failure means factual knowledge is still entangled with procedural computation.

---

# 28. Gate 5 — self-created operator / macro reuse

Enable structural creation for the first time.

The machine observes successful traces and proposes macro-operators.

A created macro is considered real only if all are true:

1. it replaces a multi-step subgraph;
2. it preserves held-out quality within a tight tolerance;
3. it reduces mean execution depth or active cost by at least a meaningful amount (target >=20% on tasks that use it);
4. it is reused across at least three distinct task/template families or held-out composition contexts;
5. disabling it causes measurable cost/quality regression;
6. it is not merely a lookup table for specific operands.

If macros appear but do not generalize beyond the traces that created them, structural learning has become memorization.

---

# 29. Gate 6 — continual skill addition

Train skill families sequentially `A -> B -> C -> D`.

Compare against:

- monolithic baseline fine-tuning;
- replay baseline;
- fixed operator library.

Measure:

- old-skill retention;
- new-skill acquisition;
- parameters/storage added;
- existing operators modified;
- interference graph.

Target:

Add a new skill primarily by adding/reusing a small subset of operators rather than rewriting the whole substrate.

A useful first pass criterion is old-skill regression under a few percentage points while new-skill quality reaches the matched baseline. Exact thresholds must be frozen before the run based on baseline noise.

---

# 30. Gate 7 — runtime and memory-system scaling

Only after Gates 1–6 show scientific value.

Measure real CPU/GPU wall-clock behavior.

Scale library storage while keeping the same active operator budget.

Report:

- operator lookup latency;
- operator load bytes;
- active compute latency;
- cache hit rate;
- state read/write bandwidth;
- end-to-end latency;
- peak memory;
- physical stored bytes.

Required engineering controls:

- dense full scan;
- hierarchical/indexed retrieval;
- grouped execution;
- cached hot operators;
- CPU-RAM cold library + GPU hot cache if useful.

If algorithmic sparsity cannot become wall-clock sparsity even after a competent fused/grouped implementation, record the hardware mismatch explicitly.

Do not invent custom hardware to rescue the result.

---

# 31. Gate 8 — tiny language transfer

Natural language begins only after the substrate passes the synthetic gates.

The first language experiment should be small enough to repeat many times.

Possible tasks:

- synthetic language with compositional grammar;
- tiny text corpus with controlled facts;
- frozen pretrained token embeddings plus ACS computation;
- distillation of selected local functions from a small pretrained model.

Do **not** immediately pretrain a 1B model.

Required held-out tests:

- unseen text corpus, not calibration text;
- held-out factual edits;
- composition/reasoning probes;
- perplexity/CE;
- generation sanity;
- route/operator usage statistics.

The held-out corpus must be part of the gate from the first language run so a calibration-local success cannot be mistaken for general language capability.

---

# 32. Gate 9 — pretrained-model bridge, only if justified

If ACS shows independent value, test conversion/assistance from an existing pretrained model.

Allowed bridges:

1. use pretrained hidden states only as inputs to operator discovery;
2. distill repeated local transforms into ACS operators;
3. identify functionally coherent low-dimensional/shared bases;
4. compile recurring dense subgraphs into reusable ACS operators;
5. use the pretrained model as a teacher, not as a hidden permanent dense solver.

The bridge fails conceptually if the final system still requires evaluating nearly the entire teacher network for every token.

---

# 33. A stronger parameter concept after foundation validation

If the substrate itself passes, revisit `taklif21`-style operator-valued parameters.

An operator may then be represented as:

`O_i(x) = Σ_{a=1..q} α_{i,a} B_a(x)`

or as a typed composition:

`O_i = B_{π_1} ∘ B_{π_2} ∘ ... ∘ B_{π_m}`.

This creates a hierarchy:

`shared primitive basis -> operator -> macro-operator -> skill`.

The purpose is not to rename many hidden scalars as “one parameter.”

Every representation must report:

- actual scalar-equivalent trainable DOF;
- stored bytes;
- active FLOPs/MACs;
- routing/index cost;
- reconstruction/functional error.

Complexity cannot be hidden behind terminology.

---

# 34. Virtual capacity target

A later goal is to create many possible operators from a much smaller physical basis.

For example:

`O_i = Compose(B, code_i)`

where `B` is a shared basis and `code_i` is a compact structural code.

If `|B| = K` and code length is `m`, the number of possible compositions can be far larger than the number physically stored.

But virtual capacity counts only if:

- different codes implement meaningfully different useful functions;
- the model can retrieve the right code;
- execution remains cheap;
- quality improves with the enlarged virtual library.

Combinatorial possibility alone is not capability.

---

# 35. Representation learning target

A foundational system must learn not only functions but useful interfaces between them.

For operator `O_i` output representation `z` and successor `O_j`, define interface compatibility loss:

`L_iface = D(O_j_input(z), target_j_state)`

or use contrastive state equivalence across programs.

The substrate should prefer canonical states that multiple operators can read.

A practical metric is cross-operator transfer:

Train `O_i` outputs with successor set `A`, then evaluate with compatible held-out successors `B`.

If every operator invents a private incompatible representation, compositional scaling will fail even if each local function is accurate.

---

# 36. Canonical-state pressure

Introduce only after a measured interface failure.

Possible regularizers:

- state normalization;
- bottleneck width;
- shared read/write projections;
- cycle consistency;
- contrastive equivalence of states with the same semantic role;
- quantized/discrete latent subspaces.

Do not add all simultaneously.

The desired property is:

`equivalent computational situations -> nearby/compatible state representations`

without forcing all content to collapse.

---

# 37. Structural exploration without combinatorial explosion

The search space of all programs is enormous.

V0 therefore uses restrictions:

1. typed composition masks;
2. Top-L candidate retrieval;
3. bounded program depth;
4. replay buffer of successful/failing traces;
5. proposal priors from structural credit;
6. macro creation only from frequent high-credit subsequences;
7. no exhaustive search over the full library.

Candidate probability may be:

`P(O_i | S_t) ∝ exp(score_i/τ) × prior_i`.

Exploration schedule:

- high entropy early;
- lower entropy after stable specialization;
- minimum exploration floor to discover alternatives.

Route collapse and route entropy must be logged.

---

# 38. Causal intervention suite

A central advantage of the substrate is that its pieces are explicit enough to intervene on.

Every major milestone should run:

1. remove operator `i`;
2. replace operator `i` with nearest neighbor;
3. shuffle descriptors;
4. freeze operator bodies but retrain controller;
5. freeze controller but retrain selected operators;
6. swap long-term memory;
7. zero working memory;
8. permute state slots where type-safe;
9. disable newly created macros;
10. replay routes with alternative compatible operators.

Claims about causality/reuse must be supported by interventions, not only route visualizations.

---

# 39. Success hierarchy

Do not call every pass a breakthrough.

## Level 0 — implementation pass

System runs correctly.

## Level 1 — local mechanism pass

One operator/routing/memory mechanism works on a controlled task.

## Level 2 — compositional pass

Known skills compose on unseen programs/depths.

## Level 3 — structural-learning pass

The system creates/reuses useful new operators or macros automatically.

## Level 4 — fixed-active scaling pass

Stored structural capacity increases capability across multiple scale points with nearly fixed active cost.

## Level 5 — language/system pass

Benefits survive held-out natural-language evaluation and real runtime measurement.

## Level 6 — new-foundation evidence

The substrate shows a reproducible Pareto improvement over strong dense/MoE baselines on **at least three** of:

- capability per active FLOP;
- compositional generalization;
- continual learning;
- editable knowledge;
- capacity scaling;
- runtime/memory efficiency.

Only Level 6 justifies language such as “new AI foundation.”

---

# 40. Kill criteria

The project must be willing to stop.

Archive or fundamentally redesign the lane if any of the following persists after the allowed rescue budget:

1. simple Gate-1 tasks require a controller/body as large as the matched dense baseline;
2. held-out composition collapses despite oracle routing, implying operator/state primitives are wrong;
3. scaling stored operator count does not improve capability under fixed active cost across multiple unsaturated tasks;
4. structural growth creates only task-specific memorized macros;
5. continual learning causes interference comparable to a monolithic baseline while using more complexity;
6. external memory cannot be swapped without broad retraining;
7. route/index cost grows nearly linearly with library size and dominates active execution;
8. gains disappear on a genuinely held-out corpus;
9. every positive result requires manually labelled operator roles at inference/training;
10. the system becomes a disguised Transformer/large dense network whose “substrate” pieces are cosmetic.

---

# 41. Rescue-budget rule

For each failed gate:

- maximum **3** architecture-level rescue attempts;
- hyperparameter sweeps are allowed only inside a predeclared narrow range and do not reset the rescue count;
- after 3 failed architecture rescues, write a failure report and move to a different foundational hypothesis.

A rescue must be motivated by measured evidence.

Bad rescue:

> “Maybe more width helps.”

Good rescue:

> “Oracle routing passes but learned routing fails; route recall is 31%, therefore test a different indexing/controller mechanism while holding operator bodies fixed.”

---

# 42. Experiment discipline

Every milestone report must contain:

- exact hypothesis;
- commit SHA;
- command/config;
- hardware;
- seed(s);
- train/eval split construction;
- active/stored parameter accounting;
- FLOPs/MACs if meaningful;
- state write bandwidth;
- route/index cost;
- quality metrics;
- held-out metrics;
- relevant baselines;
- causal ablations;
- decision: `GO`, `CONDITIONAL GO`, `NO-GO`, or `IMPLEMENTATION FAILURE`;
- exact next gate.

Raw run artifacts must be stored separately from narrative reports.

No result is promoted because it looks visually interesting.

---

# 43. Seed and statistical policy

For cheap synthetic gates:

- 1 seed may be used for implementation smoke;
- 2 seeds for provisional signal;
- at least 3 seeds for promotion;
- 5+ seeds for a claim that a noisy mechanism is stable.

For expensive language gates, fewer seeds may be acceptable, but use multiple held-out batches/corpora and bootstrap uncertainty where possible.

A result that reverses sign across seeds is not a stable pass.

---

# 44. Saturation policy

Do not test scaling on a task already at ~99–100% where improvement is impossible.

Use two benchmark lanes:

## Lane A — mechanism benchmark

Can reach ceiling; used to verify correctness.

## Lane B — capability frontier

Deliberately difficult enough that the smallest system is meaningfully below ceiling.

Only Lane B supports capacity-scaling claims.

---

# 45. Information accounting

A richer “operator” must not create fake compression by hiding scalar complexity.

Report all of:

`DOF_trainable`

`bytes_stored`

`bytes_read_per_example`

`active_MACs`

`route_MACs`

`state_bytes_written`

`execution_steps`.

If one “macroparameter” contains a million independent floats, it counts as roughly a million scalar DOF for storage/accounting.

The foundation wins only by real structure/reuse, not naming.

---

# 46. Hardware prediction before any custom hardware

The expected access pattern is:

- small hot controller/state;
- sparse reads from a large operator library;
- repeated reuse of a small working set;
- optional large external memory;
- grouped execution of operators selected by multiple tokens/examples.

This can be emulated on current hardware with:

- CPU RAM for cold operator storage;
- GPU VRAM hot cache;
- contiguous operator pages;
- grouped/batched dispatch;
- asynchronous prefetch.

Only after access statistics are measured should hardware specialization be proposed.

Potential future custom-hardware advantages would come from:

- high random-read bandwidth;
- large near-memory capacity;
- fast small-matrix/operator execution;
- cheap indirect dispatch;
- persistent state/cache.

But none of this is assumed for scientific validity.

---

# 47. What we predict may improve

These are hypotheses, not claims.

## H1 — capacity/compute decoupling

More stored reusable operators can add capability without proportional active compute.

## H2 — compositional generalization

Explicit reusable transforms and typed interfaces may compose better than distributed opaque fragments.

## H3 — continual learning

New skills may be added as new/reused operators instead of globally rewriting a monolith.

## H4 — editable knowledge

External factual memory may permit direct updates without broad retraining.

## H5 — self-compression

Frequently reused programs may compile into macro-operators, reducing execution depth.

## H6 — interpretability/diagnosability

Explicit operators and counterfactual interventions may make failure localization easier.

## H7 — hardware efficiency at scale

If locality emerges, a large cold library plus small hot working set may fit memory hierarchies better than reading all weights every token.

Any one of these can fail independently.

---

# 48. What we do NOT predict yet

Do not claim:

- replacement of Transformers;
- AGI;
- billion-parameter-equivalent capability from a tiny prototype;
- constant-time inference regardless of stored capacity;
- automatic discovery of human-level algorithms;
- zero-training knowledge insertion for arbitrary semantics;
- custom hardware speedups before measurement;
- that gradients are obsolete;
- that neural networks are unnecessary.

The first goal is only to discover whether a better **organizational mathematics** exists for some important axes.

---

# 49. First canonical implementation

Create a new independent experiment file rather than modifying the current Qwen transplant benchmark.

Suggested name:

`experiment_adaptive_computational_substrate.py`

Minimal components:

- typed state with 4–8 small slots;
- 32 small operators;
- Top-1 serial execution;
- weak query/controller;
- optional small external key-value memory;
- deterministic type mask;
- trace logger;
- counterfactual replay;
- no structural growth in the first run.

Initial operator body:

`Linear(d,d) -> activation -> Linear(d,d)`

with a bounded residual update.

Do not start with attention or a giant language encoder.

---

# 50. First milestone sequence

The autonomous research agent should execute in this order unless a gate explicitly fails.

### V0.161 (or next available milestone) — ACS implementation oracle

- deterministic operators;
- oracle routes;
- type/state/memory accounting;
- no learning claim.

### Next — learned local operators, oracle routing

Question: are the operator/state interfaces expressive and trainable?

### Next — learned routing, fixed operator library

Question: can a weak controller select useful operators without labels?

### Next — unseen depth/composition

Question: is there reusable computational structure?

### Next — fixed-active library scaling

`32 -> 128 -> 512` before attempting 2048.

### Next — memory swap/held-out address

Question: is knowledge really separate?

### Next — structural macro creation

Question: can the machine invent a reusable higher-level transform?

### Next — continual skill addition

Question: does structural locality reduce interference?

Only after these should the agent attempt tiny language transfer.

---

# 51. Decision table for the autonomous agent

| Observation | Interpretation | Next action |
|---|---|---|
| Oracle routing fails | operator/state representation wrong | redesign representation, not router |
| Oracle passes, learned routing fails | controller/addressing problem | change routing only |
| In-range passes, unseen composition fails | memorization/interface problem | test canonical state/reuse pressure |
| 32->128 improves, 128->512 flat | bounded capacity regime | diagnose utilization/expressivity before larger scale |
| More operators hurt while oracle improves | learned retrieval/credit problem | audit route recall/credit |
| Memory swap fails | knowledge entanglement | redesign memory key/interface |
| Macro lowers steps but hurts held-out quality | over-compression | reject macro creation rule |
| Continual task breaks old skills | local update not local enough | audit shared operators/controller changes |
| Algorithmic sparsity works but runtime is slow | engineering/runtime problem | fused/grouped implementation after science pass |
| Held-out language fails while calibration passes | overfit/local approximation | reject language claim immediately |

---

# 52. Relationship to existing proposals

## `taklif19` / semantic addressing

May later improve operator retrieval, but ACS must first work with a simple controller/index.

## `taklif20` / MacroCell

May later serve as one operator-body family, but giant expressive cells are not the foundation.

## `taklif21` / operator-valued parameter

Potentially powerful representation/compression layer **after** ACS demonstrates useful operator structure. It may be especially natural for shared operator bases.

## `taklif22` / Latent Computational Machine

Closest relative. `taklif22` is primarily an architecture decomposition into computation, memory, working state, and semantic control. `taklif23` goes one level deeper: it treats the **operator library and execution structure themselves as learned/evolving mathematical objects**, separates local parameter learning from structural credit, defines operator creation/merge/split/prune, and changes the scaling unit from parameter count toward reusable computational structure.

Therefore `taklif23` does not supersede `taklif22`; it is the foundational lane that asks whether the decomposition can become a self-organizing learning substrate rather than a fixed architecture.

---

# 53. Strongest possible positive result

The high-value result is not merely higher accuracy on one synthetic task.

The strongest early evidence would look like:

1. a 32-operator system learns a set of primitive skills;
2. the same operators compose on unseen depth/programs;
3. growing the stored library `32 -> 128 -> 512 -> 2048` improves held-out capability while active operator count remains almost fixed;
4. new factual memory can be swapped without retraining the operator library;
5. the machine automatically compiles repeated useful subgraphs into reusable macros that reduce execution cost;
6. new skill families can be added with little damage to old skills;
7. a tiny language transfer preserves at least some of these advantages on genuinely held-out text.

If several of these hold simultaneously, the project has evidence for a genuinely different learning organization, not merely another sparse neural architecture.

---

# 54. Strongest possible negative result

A valuable negative outcome is also clear.

If:

- oracle programs work but learned structural credit cannot scale;
- or operators require private incompatible states;
- or capacity growth does not improve unsaturated held-out tasks;
- or structural search cost explodes;
- or language transfer immediately requires restoring a large dense model;

then the project should conclude that this particular substrate does not beat the fixed differentiable-network organization and stop investing in it.

That conclusion would still be useful because it isolates **why** the more radical foundation failed.

---

# 55. Final research contract

`taklif23` is accepted only under the following contract:

1. **No ideology.** Gradients, neural modules, symbolic operations, search, and memory are tools, not identities.
2. **No hidden complexity.** Count real DOF, bytes, FLOPs, routing, and state traffic.
3. **No calibration-only victories.** Held-out evaluation is mandatory from the start of every serious gate.
4. **No giant-model escape.** Small experiments must reveal the mechanism before scale-up.
5. **No endless patching.** Three architecture rescues per failed gate maximum.
6. **No conflation.** Operator quality, routing quality, memory quality, structural credit, and runtime are tested separately.
7. **No scaling claim from one jump.** Multiple capacity points and seeds are required.
8. **No custom-hardware excuse.** Software first.
9. **No “macroparameter” accounting tricks.** Rich objects must expose their true cost.
10. **Promote only causal mechanisms.** Ablations/interventions must show which component produced the gain.

The question this proposal asks is deliberately fundamental:

> **Can an AI system scale by accumulating, composing, editing, and reusing learned computational structures—while treating dense scalar optimization as a local implementation tool rather than the global definition of intelligence?**

If the answer is yes even on small but carefully controlled gates, continue aggressively. If not, archive the lane early and preserve the evidence.