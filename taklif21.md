# taklif21.md — Structured / operator-valued parameters: replacing scalar weights with learned algebraic transforms

Status: **canonical independent research proposal; high-risk/high-upside; not yet validated**

> **HARD ISOLATION RULE:** Do **not** combine this proposal with `taklif19.md` or `taklif20.md` MacroCells in the first experiment. Do not change cell granularity, routing architecture, memory architecture, and parameter algebra simultaneously. `taklif21` must first be tested as an isolated parameterization change against matched controls. MacroCell + operator-valued parameter integration is forbidden until both directions have independent evidence.

---

# 0. Motivation and central hypothesis

Conventional neural networks ultimately store trainable relationships as scalar real numbers. A linear layer is

`y = W x + b`,

where each elementary connection is a scalar `w_ij ∈ R`.

The intuitive proposal is to make an elementary learned relationship richer than one scalar. However, simply packing 100 independent floats into one object and calling it "one large parameter" is fake compression: the system still has 100 trainable scalar degrees of freedom (DOF).

The scientifically meaningful hypothesis is instead:

> A small number of trainable coefficients can select/combine a shared set of learned structured operators, so an elementary connection behaves as a rich local transform while using far fewer independent coefficients than an unconstrained transform.

The canonical form is

`Θ_e = Σ_{a=1..q} α_{e,a} B_a`,

where:

- `e` indexes one packet-to-packet connection/block;
- `α_e ∈ R^q` is the connection-specific coefficient vector;
- `B_a ∈ R^(g×g)` are globally or layer-shared learned basis operators;
- `g` is packet/state width;
- `q << g^2`.

Instead of a scalar connection

`x -> w x`,

we obtain a structured local transform

`x_packet -> Θ_e x_packet`.

The high-level hypothesis is that learned shared operator structure may improve the **capability / scalar-DOF / active-compute frontier**, especially when useful transformations recur across many connections.

This is not a theorem. It can fail because the useful weight space may not lie near a low-dimensional shared operator span, optimization may become ill-conditioned, or the structured implementation may be slower than an ordinary dense matrix.

---

# 1. Terminology: what counts as a parameter

This proposal uses four different quantities. They must never be conflated.

## 1.1 Logical parameter object

One structured object `Θ_e` may be called one **operator parameter object** at the software/architecture level.

This count is descriptive only and must never be used to claim compression.

## 1.2 Trainable scalar DOF

The real parameter count for scientific comparison is the number of independent trainable scalar values.

If

`Θ_e = Σ_a α_{e,a} B_a`,

then both the coefficients `α` and all learned basis entries count.

## 1.3 Materialized effective matrix size

`Θ_e` may behave as a `g×g` matrix containing `g^2` effective entries. These effective entries are not independent when `q << g^2`.

Do not call `g^2` the trainable parameter count.

## 1.4 Runtime compute and memory traffic

A representation with fewer scalar DOF can still use more FLOPs, gather operations, memory bandwidth, or kernel launches. Therefore report separately:

- scalar trainable DOF;
- stored bytes;
- effective matrix dimensions;
- MAC/FLOP estimate;
- peak VRAM/RAM;
- latency;
- throughput.

A proposal that only wins by renaming parameters is a **NO-GO**.

---

# 2. Canonical packetized layer

Let an ordinary hidden vector be

`x ∈ R^(d_in)`.

Choose a packet width `g` such that `g | d_in` and `g | d_out` for the first clean experiment.

Define

`n_in = d_in / g`,

`n_out = d_out / g`.

Reshape

`x = [x_1, ..., x_n_in]`, with `x_i ∈ R^g`.

For every output packet `o` and input packet `i`, define

`Θ_{o,i} = Σ_{a=1..q} α_{o,i,a} B_a`.

Then

`y_o = Σ_{i=1..n_in} Θ_{o,i} x_i + b_o`.

Expanded:

`y_o = Σ_i Σ_a α_{o,i,a} B_a x_i + b_o`.

This is a block-structured linear transform in which every `g×g` block lies in the learned span

`S_B = span{B_1, ..., B_q}`.

The ordinary full matrix allows each block to occupy all of `R^(g×g)`. The structured layer restricts each block to a shared `q`-dimensional subspace.

---

# 3. Canonical V1 numbers

The first experiment must use a deliberately simple configuration:

- packet width `g = 16`;
- number of shared basis operators `q = 8`;
- basis operators are dense `16×16` matrices in V1;
- one shared basis bank per tested layer or layer-group, not per connection;
- connection-specific trainables are only the 8 coefficients `α_{o,i,:}`;
- no MacroCells;
- no new router;
- no recurrent memory change;
- no factorized virtual circuit change in the same run.

Each unconstrained `16×16` block contains

`g^2 = 256`

independent weights.

Each structured block contains only

`q = 8`

connection-specific coefficients, plus amortized shared bases.

For many blocks, the asymptotic connection-specific reduction is

`256 / 8 = 32×`.

This is **not** a 32× whole-layer compression claim because the learned bases also count and compute may not fall by 32×.

---

# 4. Exact scalar-DOF accounting

Let

`E = n_out * n_in`

be the number of packet-to-packet blocks.

## 4.1 Conventional full block matrix

Ignoring bias:

`P_full = E g^2`.

## 4.2 Structured operator parameterization

Connection coefficients:

`P_coeff = E q`.

Shared dense bases:

`P_basis = q g^2`.

Total:

`P_struct = E q + q g^2`.

Compression ratio:

`R_P = P_full / P_struct`

`= E g^2 / (E q + q g^2)`.

For `E >> g^2`,

`R_P ≈ g^2 / q`.

For canonical `g=16`, `q=8`, the large-layer asymptote is approximately `32×` fewer scalar weights for that isolated matrix parameterization.

Again: this is an algebraic parameter-count result, not a quality or speed result.

---

# 5. Important equivalence: this is a constrained block matrix, not magic

The whole structured layer can always be materialized as an ordinary matrix `W_struct`.

Therefore this proposal does **not** create a function outside ordinary neural networks. Instead it imposes a specific structured parameter manifold.

Let

`A_a ∈ R^(n_out×n_in)`

contain coefficients `α_{o,i,a}` for basis `a`.

Then the full matrix can be written as a sum of Kronecker products:

`W_struct = Σ_{a=1..q} A_a ⊗ B_a`.

This identity is important.

The proposal is therefore equivalent to learning a low Kronecker-rank decomposition of a large weight matrix, with both macro coefficient matrices `A_a` and local operators `B_a` trainable.

This gives a precise mathematical interpretation and prevents mystical claims about "larger parameters".

Core question:

> Does this structured manifold contain useful solutions that SGD can find more efficiently/generalizably than an equally budgeted conventional parameterization?

---

# 6. Expressivity

## 6.1 Fixed-basis block expressivity

For fixed linearly independent bases `B_a`, each local block lies in a subspace of dimension at most `q`:

`dim(S_B) ≤ q`.

If `q < g^2`, not every `g×g` matrix can be represented.

Therefore structured parameters trade unrestricted local expressivity for reuse/regularity.

## 6.2 Full expressivity limit

If `q = g^2` and `{B_a}` spans the standard matrix basis, any block can be represented exactly.

Thus the constrained architecture approaches an ordinary dense block matrix as `q -> g^2`.

## 6.3 Approximation error

For a target block `W_e*`, the best fixed-basis approximation is

`W_hat_e = argmin_{W ∈ S_B} ||W_e* - W||_F`.

If the basis is orthonormal under the Frobenius inner product,

`α_{e,a}* = <W_e*, B_a>_F`

and

`||W_e* - W_hat_e||_F^2`

is the energy outside the learned basis span.

This gives a direct diagnostic: after training or when compressing a teacher, measure how much teacher block energy is captured by the top `q` learned/shared operator directions.

If residual energy stays high even with sensible `q`, the hypothesis is representation-limited.

---

# 7. Identifiability and gauge symmetry

The decomposition is not unique.

For any invertible `T ∈ R^(q×q)`, transformed bases and coefficients can represent the same `W`:

`B'_a = Σ_b T_{ab} B_b`,

`α'_e = T^{-T} α_e`

(up to indexing convention).

Therefore individual basis operators are generally **not identifiable** without extra constraints.

Consequences:

- do not interpret one learned `B_a` as a unique semantic primitive merely because it has a visual pattern;
- basis drift/rotation can make training diagnostics misleading;
- coefficient sparsity can be basis-dependent;
- comparing basis index `a` across seeds may be meaningless.

Canonical stabilization options, applied only if needed:

1. Frobenius normalization `||B_a||_F ≈ 1`;
2. soft orthogonality penalty
   `L_orth = ||G - I||_F^2`, where `G_ab = <B_a,B_b>_F`;
3. coefficient norm control.

Do not impose strong orthogonality by default if it hurts useful correlated operators.

---

# 8. Scale degeneracy

There is a simple scaling symmetry:

`B_a -> c B_a`,

`α_{e,a} -> α_{e,a}/c`.

The represented weight is unchanged.

Without control, basis norms may explode while coefficients shrink, or vice versa, causing poor conditioning and optimizer instability.

Required monitoring:

- `||B_a||_F` distribution;
- `||α_e||_2` distribution;
- gradient norms for basis vs coefficient tensors;
- effective matrix spectral norms.

Possible stabilization:

`B_a = s_a * normalize(B_tilde_a)`

with explicit learned scale `s_a`, or periodic/forward normalization.

This is a likely failure point.

---

# 9. Forward computation and hardware cost

Naively materializing every block

`Θ_{o,i} = Σ_a α_{o,i,a} B_a`

before multiplying by `x_i` may destroy all efficiency.

Use the algebraic form

`y_o = Σ_a Σ_i α_{o,i,a} (B_a x_i)`.

There are two natural contraction orders.

## 9.1 Basis-first

For each input packet and basis:

`z_{i,a} = B_a x_i`.

Then mix across packet graph:

`y_o = Σ_a Σ_i α_{o,i,a} z_{i,a}`.

Approximate basis-transform MAC cost:

`C_basis ≈ n_in q g^2`.

Coefficient mixing cost:

`C_mix ≈ E q g`.

Total:

`C_struct ≈ n_in q g^2 + E q g`.

Conventional dense block matrix cost:

`C_full ≈ E g^2`.

For large `E`, ratio is approximately

`C_struct / C_full ≈ q/g`

plus basis overhead.

Canonical `q=8`, `g=16` suggests an asymptotic arithmetic ratio around `0.5`, not `1/32`.

This distinction is critical:

> Parameter compression can be ~32× while arithmetic compression may be only ~2×.

Real latency may be worse or better depending on kernels, tensor layout, cache reuse, and batching.

## 9.2 Coefficient-first

Alternative reorderings may mix packets before applying bases. Benchmark both mathematically equivalent contractions if shapes permit.

No speed claim is accepted without actual wall-clock measurement.

---

# 10. Optional low-rank basis extension — forbidden in the first gate

Later, a basis may be factorized

`B_a = U_a V_a^T`,

with rank `r_b << g`.

Then

`P_basis_lowrank = 2 q g r_b`

instead of `q g^2`, and application cost becomes approximately `2 q g r_b` per packet.

However this changes two variables at once: operator-valued parameterization and basis rank.

Therefore **do not use low-rank `B_a` in the first canonical test**. Only test it after dense-basis V1 establishes a useful quality/DOF frontier.

---

# 11. Relation to scalar parameters

A scalar weight is a special case.

If `g=1`, each `B_a` is scalar and the construction collapses to an ordinary linear parameterization.

Alternatively, for `g>1`, if the only basis is identity

`B_1 = I`,

then

`Θ_e = α_e I`,

which acts like one scalar shared across all coordinates of a packet.

Therefore the structured proposal strictly generalizes scalar-like packet scaling when multiple nontrivial bases are allowed.

This is an expressivity relation only; it does not guarantee better learning.

---

# 12. Why this may help Neural Engine

The project has repeatedly encountered a tension between very small computational primitives and distributed representations.

Structured operator parameters may help in four ways.

## 12.1 More local transformation per learned connection

One coefficient packet can select a structured `g×g` transform rather than independently tuning `g^2` unrelated scalar entries.

## 12.2 Reuse

The same operator bases receive gradient from many packet connections, potentially learning reusable transformation primitives.

## 12.3 Statistical regularization

Restricting each block to a shared span may reduce overfitting and improve OOD behavior if the task truly contains repeated algebraic structure.

## 12.4 Better growth geometry

A learned basis can potentially be reused when width/bank capacity grows, analogous in spirit to the project's successful reusable-factor direction.

These are hypotheses. Shared structure can also be harmful if different connections genuinely need unrelated transforms.

---

# 13. What this proposal is NOT

It is not:

- a claim that one operator object counts as one scientific parameter;
- a MacroCell;
- MoE routing;
- a new activation function;
- a proof of compression;
- a proof of new mathematical intelligence;
- permission to hide scalar weights inside a custom class and report a smaller count;
- a replacement for wall-clock benchmarks.

All trainable scalar values must be counted.

---

# 14. Training objective

For a standalone supervised/student model:

`L_task` = normal task or LM loss.

For teacher transfer:

`L = λ_task L_task + λ_out L_out + λ_hidden L_hidden + optional regularizers`.

Possible components:

`L_out = ||f_struct(x) - f_teacher(x)||^2`

or logit KL for a language model.

Optional basis regularization:

`L_orth = ||G-I||_F^2`.

Optional coefficient regularization:

`L_alpha = E_e ||α_e||_2^2`.

Canonical first run should use the smallest useful loss set. Do not activate orthogonality, sparsity, entropy, coefficient L1, spectral penalties, and distillation tricks simultaneously.

Recommended sequence:

1. task/distillation loss only;
2. inspect conditioning and basis collapse;
3. add only the minimum stabilization required;
4. rerun matched seeds.

---

# 15. Initialization

Poor initialization can create an unfair negative result.

Two valid initializations should be tested.

## 15.1 Random structured initialization

Initialize `B_a` with variance preserving packet activation scale and `α` with variance such that the effective block variance matches a conventional layer.

If entries of `B_a` have variance `σ_B^2` and coefficients variance `σ_α^2`, then approximately

`Var[(Θ_e)_{uv}] ≈ q σ_α^2 σ_B^2`

assuming independence/zero mean.

Choose the product so the effective `W_struct` has the same target variance as Xavier/He initialization for the matched layer.

## 15.2 Teacher decomposition initialization

For replacing a pretrained dense matrix, reshape the teacher matrix into blocks and fit a rank-`q` shared basis decomposition before end-to-end training.

One practical initialization is alternating least squares / SVD-like factorization over vectorized blocks.

Let each block be flattened:

`w_e = vec(W_e*) ∈ R^(g^2)`.

Stack rows into

`M ∈ R^(E × g^2)`.

Rank-`q` truncated SVD

`M ≈ U_q Σ_q V_q^T`

provides:

- basis vectors from rows of `V_q^T`, reshaped into `B_a`;
- connection coefficients from `U_q Σ_q`.

This gives the **best rank-q approximation in Frobenius norm** for the block matrix dataset under the linear shared-span model.

This is an extremely important oracle/upper-bound diagnostic.

If even teacher-SVD initialization has large functional error at useful `q`, scratch training is unlikely to magically remove the representational bottleneck.

---

# 16. Teacher reconstruction oracle

Before expensive training, measure the teacher weight reconstruction frontier.

For each candidate `q ∈ {2,4,8,16,32,...}`:

1. block the teacher weight into `g×g` matrices;
2. compute rank-`q` SVD over vectorized blocks;
3. reconstruct;
4. report normalized Frobenius error
   `ε_W = ||W* - W_hat||_F / ||W*||_F`;
5. run the actual layer/model with reconstructed weights;
6. report output error / CE / KL.

This separates two failure modes:

- representation cannot approximate teacher even with optimal coefficients;
- representation is sufficient, but SGD/training fails to find it.

Do not waste overnight routing/training sweeps if the reconstruction oracle already shows a hard representation ceiling.

---

# 17. Canonical experiments

Run in increasing cost order.

## Experiment A — synthetic linear recovery

Generate target matrices with known shared operator structure and verify the implementation can recover them.

Positive control:

`W_target = Σ_a A_a* ⊗ B_a*`.

The model should fit to numerical precision when `q` is sufficient.

Negative/control target:

random unconstrained dense blocks with no low-rank shared structure.

The structured model should show the expected approximation ceiling at small `q`.

If it fits arbitrary random targets suspiciously well with too few DOF, inspect for a parameter-count/implementation leak.

## Experiment B — current Neural Engine synthetic tasks, parameterization only

Replace a selected conventional linear transform with operator-valued parameterization while keeping:

- same data;
- same router;
- same micro-cells;
- same register graph;
- same training steps/tokens;
- no MacroCell.

Compare matched scalar-DOF and matched-compute controls.

## Experiment C — Qwen layer reconstruction

Use real pretrained Qwen matrices as teacher targets without changing Qwen architecture.

Test whether `g=16,q=8` captures useful weight/output structure.

Start with selected late FFN projection matrices or another isolated matrix where replacement can be audited cleanly.

Report both matrix reconstruction and full-model fidelity.

## Experiment D — trained structured student

If C is promising, train the structured replacement using teacher hidden/output losses and then end-to-end fidelity.

Only after these gates consider broader replacement.

---

# 18. Required baselines

At least the following controls are required where practical.

## B0 — ordinary dense matrix

Same architecture/task, conventional parameterization.

## B1 — parameter-matched ordinary low-rank factorization

For example `W = UV^T` with scalar trainable DOF approximately matched to `P_struct`.

This is essential. Otherwise any gain may simply come from low-dimensional regularization rather than operator-valued structure.

## B2 — block-diagonal / grouped linear control

Match packetization without shared operator basis to determine whether benefits come merely from grouping.

## B3 — fixed random operator bases

Freeze random `B_a`, train only `α`.

If learned bases do not beat fixed random bases, the claimed learned algebra may not matter.

## B4 — SVD/PCA teacher basis

For teacher compression, compare learned end-to-end bases with best weight-space SVD initialization.

## B5 — equal scalar DOF conventional MLP/linear student

Ensures that performance is not simply due to giving the structured model more trainable values.

## B6 — equal real FLOP/latency control

A structured model with fewer parameters but twice the execution cost is not an efficiency win unless quality gain justifies it.

---

# 19. Fairness axes

One comparison cannot match every resource simultaneously, so report at least three frontiers.

## 19.1 Equal scalar-DOF frontier

`Q vs trainable scalar DOF`.

## 19.2 Equal theoretical compute frontier

`Q vs MAC/FLOP estimate`.

## 19.3 Equal measured runtime frontier

`Q vs actual latency/throughput` on the same hardware/software stack.

Also log stored bytes and memory traffic.

A result is strongest if it improves more than one frontier.

---

# 20. Main failure modes

## 20.1 Fake parameter compression

Symptom: report says "one large parameter" while the object contains hundreds of independent trainable floats.

Prevention: always count scalar DOF.

## 20.2 Basis collapse

Several `B_a` become nearly identical, reducing effective rank.

Audit Gram matrix singular values and effective basis rank.

## 20.3 Basis over-dispersion

Bases are forced orthogonal even when task wants correlated transforms; regularization harms quality.

Use orthogonality only when collapse is observed.

## 20.4 Scale blow-up

`||B||` grows while `||α||` shrinks or vice versa.

Audit norms and effective spectral scales.

## 20.5 Poor conditioning

The basis Gram matrix becomes ill-conditioned, producing unstable coefficient gradients.

Log condition number/eigenvalues of `G`.

## 20.6 Representation ceiling

Useful target blocks do not lie near a low-q shared span.

SVD reconstruction oracle exposes this early.

## 20.7 Layer heterogeneity

Different layers may require different bases. A single global basis bank may be too restrictive.

Canonical V1 uses layer-local or small layer-group sharing first. Global sharing is a later hypothesis, not assumed.

## 20.8 Packet-boundary artifact

Arbitrary grouping of channels into contiguous packets may destroy useful interactions or make results basis-order dependent.

Required ablations:

- contiguous packets;
- fixed random permutation before grouping;
- possibly learned permutation only after first gates.

Do not introduce learned routing/permutation immediately.

## 20.9 Basis permutation / gauge ambiguity

Different seeds can learn rotated equivalent bases. Do not judge semantic reproducibility by basis index equality.

Compare represented subspaces/principal angles or effective functions.

## 20.10 Gradient interference through shared bases

Because many connections update the same `B_a`, unrelated gradients may conflict.

Audit:

- basis-gradient cosine statistics across batches/tasks/layers;
- basis update norms;
- whether coefficients absorb all specialization while bases remain generic.

## 20.11 Shared-basis bottleneck

Too much sharing may underfit; increasing `q` can help but also destroys compression.

Report full quality-vs-q frontier, not one cherry-picked value.

## 20.12 q growth degenerates to dense

As `q -> g^2`, the model approaches unconstrained blocks. If quality only recovers near `q ≈ g^2`, the hypothesis has failed to provide useful structural compression.

## 20.13 Hardware mismatch

Kronecker/basis contractions may benchmark poorly on GPU despite fewer FLOPs because standard GEMM is extremely optimized.

Measure wall clock. Do not infer speed from equations alone.

## 20.14 Materialization overhead

Constructing full `Θ` matrices each step can erase benefits.

Use fused/reordered contractions and benchmark both.

## 20.15 Numerical precision

Shared basis sums can have different accumulation order and dynamic range from dense GEMM. Test float32 correctness before fp16/bf16 claims.

## 20.16 Optimizer imbalance

`B` tensors are globally shared while `α` tensors are local; they may require different learning-rate scales or optimizer statistics.

First try one optimizer with gradient/norm logging. Split learning rates only if evidence shows imbalance.

## 20.17 Coefficient memorization

`α` may simply memorize connection identity while shared bases provide little useful structure.

Compare learned-basis model against random frozen bases and analyze effective basis contribution.

## 20.18 Basis memorization

With too many layer-specific bases, the model can hide the original dense matrix inside bases and defeat the purpose.

Count all scalar DOF and enforce the intended sharing scope.

## 20.19 OOD false positive

Parameter regularization can improve one narrow OOD split by chance. Use multiple OOD forms and seeds.

## 20.20 Training-budget confound

Structured models may converge slower/faster. Report both equal-step/token and, where needed, convergence-matched curves.

---

# 21. Diagnostics that must be logged

Every run should record:

- `g`, `q`, sharing scope;
- scalar trainable DOF;
- effective/materialized matrix size;
- basis count and effective rank;
- Gram eigenvalues/condition number;
- basis Frobenius/spectral norms;
- coefficient norm distribution;
- reconstruction error if teacher exists;
- task quality / CE / KL / top-1 as appropriate;
- theoretical MACs;
- measured latency and throughput;
- VRAM/RAM;
- kernel implementation path;
- seed;
- training steps/tokens;
- GPU-hours;
- packet permutation/grouping scheme.

For pretrained transfer also log teacher/student hidden/output error by layer so accumulation can be localized.

---

# 22. Granularity sweep

Do not optimize dozens of configurations immediately.

Canonical initial sweep:

- `g=8, q=4`;
- `g=16, q=8` **primary**;
- `g=32, q=8 or 16` only if first two justify it.

The useful quantity is roughly the structural compression ratio `g^2/q`, but compute ratio behaves differently (`q/g` in the large-block basis-first approximation).

This creates a tradeoff:

- larger `g` gives richer local operators and stronger parameter sharing;
- larger `g` can create harder optimization, worse packet boundaries, and expensive basis transforms;
- larger `q` increases expressivity but reduces compression and compute benefit.

Do not sweep more than these coarse points before the primary hypothesis has evidence.

---

# 23. Optional coefficient sparsity — NOT part of V1

One may later encourage only a few basis operators per connection:

`||α_e||_0 << q`

or use top-k/gated coefficients.

This could reduce compute further and create interpretable local operator composition.

However it introduces another routing/sparsity problem and must not be mixed into V1.

V1 coefficients are dense over `q=8`.

---

# 24. Optional learned algebra closure — later research only

A more radical extension is to learn approximate composition constants

`B_i B_j ≈ Σ_k c_{ijk} B_k`.

This would define an emergent finite-dimensional operator algebra with structure tensor `c_{ijk}`.

Potential closure loss:

`L_close = Σ_{i,j} ||B_i B_j - Σ_k c_{ijk} B_k||_F^2`.

This is scientifically interesting but **forbidden in the first proposal21 gate** because it adds a second strong hypothesis.

Only consider it if ordinary learned operator bases already show clear value.

---

# 25. Relation to factorized virtual circuits

The project already uses reusable factor rows to create many virtual circuits. Operator-valued parameters have conceptual overlap: both replace many independent weights with shared reusable structure plus small local codes.

This overlap is a reason for interest, but also a confound.

Therefore:

- first test operator-valued parameters in an otherwise unchanged model;
- do not immediately replace current factorized circuit internals;
- if both independently work, later compare additive/complementary benefit.

A combined win must exceed either technique alone under equal scalar DOF and compute.

---

# 26. Relation to MacroCells — strict prohibition in first study

`taklif19`/`taklif20` change the **computational unit granularity**.

`taklif21` changes the **parameter algebra inside an otherwise fixed computation graph**.

They answer different scientific questions.

First-study rule:

```text
MacroCell changes:            OFF
operator-valued parameters:   ON
router changes:               OFF
new memory graph:             OFF
```

After independent results exist, a separate future proposal may test:

`MacroCell + operator-valued parameters`.

Do not infer that combination is good merely because either component works alone.

---

# 27. GO / NO-GO gates

## Gate 0 — implementation correctness

Positive synthetic structured target must be recovered to numerical tolerance when representable.

Failure => implementation/optimization must be fixed before scientific conclusions.

## Gate 1 — teacher representation oracle

At a useful small `q`, weight/output reconstruction must be materially better than trivial/random structured controls.

If useful quality requires `q` near `g^2`, **NO-GO for structural compression**.

## Gate 2 — equal-DOF quality

At matched trainable scalar DOF, structured parameters must beat or clearly match ordinary low-rank/dense controls on at least one nontrivial capability/generalization metric.

If ordinary low-rank factorization performs equally well or better, operator-valued structure has not justified itself.

## Gate 3 — OOD/composition

Any claimed representational advantage should survive harder/OOD evaluation, not only train-set or saturated arithmetic accuracy.

## Gate 4 — runtime

For an efficiency claim, optimized structured execution must show competitive or better wall-clock behavior. Fewer scalar weights alone is insufficient.

## Strong GO

A strong result would show, reproducibly across seeds:

- materially fewer scalar DOF;
- comparable or better difficult/OOD quality than matched conventional controls;
- learned bases outperform fixed random bases;
- useful `q << g^2`;
- stable conditioning;
- no hidden materialization/compute explosion;
- real runtime/memory benefit or a clearly superior quality-per-resource frontier.

## NO-GO

Stop this direction if, after the canonical sweep:

- SVD/teacher oracle says low-q representation is fundamentally inadequate;
- quality only recovers when `q` approaches `g^2`;
- learned bases do not beat random/fixed bases;
- ordinary low-rank factorization matches or beats it at equal scalar DOF;
- basis training is persistently unstable across reasonable normalization/initialization;
- hardware cost overwhelms all parameter savings;
- benefits disappear on independent/OOD tests.

Do **not** respond to NO-GO by endlessly increasing `g`, `q`, basis hierarchy, sparsity, learned permutations, algebra closure, or MacroCell complexity.

---

# 28. Recommended execution order for the autonomous agent

1. Implement generic packetized operator-valued linear layer with exact shape tests.
2. Verify `q=g^2`/standard-basis construction can reproduce an arbitrary block matrix as a correctness control.
3. Verify synthetic low-operator-rank targets can be recovered at small `q`.
4. Add scalar-DOF/MAC/latency accounting tests.
5. Build teacher block-SVD reconstruction oracle.
6. Test canonical `g=16,q=8` on an isolated real/pretrained matrix/layer.
7. Compare ordinary low-rank, grouped/block, fixed-random-basis, and learned-basis controls.
8. Only if promising, train the structured student end-to-end on a small controlled target.
9. Run OOD/composition and independent text controls.
10. Benchmark real GPU runtime with no full-Θ materialization.
11. Freeze result as GO/NO-GO before any extension.

**Do not execute `taklif19`/`taklif20` MacroCell integration inside this sequence.** If the overnight agent is processing proposals sequentially, proposal21 must be evaluated and frozen independently.

---

# 29. Claims discipline

Allowed if measured:

- "A `g=16,q=8` structured layer used X scalar trainable parameters."
- "It represented each 16×16 block through an 8-dimensional learned operator span."
- "It matched/outperformed baseline Y at equal scalar DOF on benchmark Z."
- "Measured latency was X vs Y on hardware H."

Not allowed without evidence:

- "One parameter contains 256 times more knowledge."
- "We reduced model parameters 32×" when bases/other layers are omitted from accounting.
- "Operator parameters are more intelligent."
- "This creates a new algebra" unless algebraic structure/closure is explicitly trained and demonstrated.
- "It is faster" based only on theoretical parameter count.

---

# 30. Final research question

The proposal reduces to one falsifiable question:

> **Can a neural model replace many unrelated scalar/block weights with a small learned basis of reusable local operators plus cheap connection-specific codes, and thereby obtain a better quality/generalization/resource frontier than conventional parameterizations?**

Canonical first test:

`g = 16`, `q = 8`, dense learned basis operators, no MacroCells, no routing changes, all scalar DOF counted, teacher-SVD oracle first, matched ordinary low-rank and fixed/random-basis controls, followed by real latency measurement.

If this passes, later research may explore low-rank basis operators, coefficient sparsity, global/shared operator vocabularies, learned algebra closure, or combination with MacroCells. None of those are part of the first gate.
