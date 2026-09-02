# taklif20.md — Mesoscopic MacroCells: equal-active-budget large-cell architecture and falsification plan

Status: **canonical high-priority research proposal; not yet validated**

This proposal tests a specific version of the "fewer but much stronger cells" hypothesis. It is intentionally separated from `taklif19.md`:

- `taklif19.md` asks how cells can be self-describing, semantically addressed, and sparsely routed.
- `taklif20.md` asks a more basic architectural question: **at the same active parameter budget, is it better to execute many tiny micro-circuits or only one/two much richer computational cells?**

The proposal is high priority because it directly targets four observed failure modes of the current Neural Engine: routing fragmentation, long micro-composition depth, distributed credit assignment across tiny circuits, and the failure of pretrained Qwen neuron chunks to become useful 4x sparse primitives.

It must remain falsifiable. If large cells do not improve the capability/active-compute frontier under matched budgets, do not keep increasing cell size indefinitely.

---

# 0. Core hypothesis

Let a current computation require a sequence of small routed transforms

`F(x) ≈ C_{i_T} ∘ ... ∘ C_{i_2} ∘ C_{i_1}(x)`.

If each `C_i` is weak, the model must solve a difficult discrete planning problem over many routing decisions. A stronger cell may absorb a useful local program:

`F(x) ≈ M_{j_s} ∘ ... ∘ M_{j_1}(x)`

with `s << T`, ideally `s=1..2` for many local transformations.

The central test is therefore not simply parameter count. It is whether increasing **computation granularity** reduces routing/planning depth and improves generalization while keeping active compute approximately fixed.

Formally, define the capability frontier

`Q(B_active, N_stored, D_route)`

where:

- `B_active` = active trainable/body parameters executed per step;
- `N_stored` = number of stored computational units;
- `D_route` = number of routing decisions / effective routed depth;
- `Q` = task quality on difficult in-distribution and OOD evaluations.

The hypothesis predicts that for some mesoscopic cell size `P_cell*`,

`Q_macro(B_active) > Q_micro(B_active)`

because larger cells reduce `D_route` and improve coherent local function learning without becoming giant generalist experts.

This is an empirical hypothesis, not a theorem.

---

# 1. Baseline micro-circuit size

The current simplified low-rank micro-circuit is

`C_i(x) = U_i φ(D_i x) + b_i`

with state dimension `d` and low rank `r`.

Ignoring small implementation details, parameter count is

`P_micro = d*r + r*d + d = 2dr + d`.

For the current common values

- `d = 384`
- `r = 16`

we obtain

`P_micro = 2*384*16 + 384 = 12,672` parameters.

This is the reference unit for the size ratio below.

---

# 2. Canonical mesoscopic MacroCell V1

The V1 cell must be richer than a low-rank MLP, but it must not become an entire Transformer/MoE expert. It has exactly four functional components:

1. local/state transform;
2. memory/register transform;
3. multiplicative/bilinear interaction;
4. residual output and memory write.

Let:

- `x ∈ R^d` = local/current state;
- `r ∈ R^d` = selected/pooled memory-register read;
- `h` = MacroCell hidden width;
- `b` = bilinear latent rank.

Define

`z_x = W_x x`

`z_m = W_m r`

`p_x = P_x x`

`p_m = P_m r`

`z_b = W_b (p_x ⊙ p_m)`

`u = z_x + z_m + z_b + β`

`h_c = φ(u)`

`y = W_o h_c`

`Δm = W_w h_c`.

The body therefore implements

`M_i(x,r) = (y_i, Δm_i)`.

For serial execution:

`x_{t,0} = x_t`

`(y_{t,j}, Δm_{t,j}) = M_{i_j}(x_{t,j-1}, r_{t,j-1})`

`x_{t,j} = x_{t,j-1} + α_{t,j} y_{t,j}`

`m_{t,j} = Update(m_{t,j-1}, α_{t,j} Δm_{t,j})`.

After `k` active cells:

`x_{t+1} = x_{t,k}`

`m_{t+1} = m_{t,k}`.

Canonical V1 has no internal attention, no inner router, no recursion, and only one nonlinear hidden stage. This prevents a false positive caused by hiding a whole model inside each cell.

---

# 3. Exact equal-active-budget derivation

The current factorized 500M/700M research line reports an active parameter estimate of approximately

`B_ref = 1,792,305`.

We want `k=2` active MacroCells while preserving essentially the same active body budget.

For the V1 equations above, with `d_m=d`, parameter count is

`P_cell = 4dh + 2bd + hb + h`

because:

- `W_x`: `h*d`
- `W_m`: `h*d`
- `P_x`: `b*d`
- `P_m`: `b*d`
- `W_b`: `h*b`
- bias: `h`
- `W_o`: `d*h`
- `W_w`: `d*h`.

Choose

- `d = 384`
- `h = 480`
- `b = 128`.

Then

`4dh = 4*384*480 = 737,280`

`2bd = 2*128*384 = 98,304`

`hb = 480*128 = 61,440`

`h = 480`.

Therefore

`P_cell = 737,280 + 98,304 + 61,440 + 480 = 897,504`.

With `k=2`:

`B_macro = 2*897,504 = 1,795,008`.

Relative difference from the current reference active estimate:

`(1,795,008 - 1,792,305) / 1,792,305 ≈ 0.001508 ≈ 0.151%`.

Thus the primary comparison is almost exactly active-parameter matched.

One MacroCell corresponds to

`897,504 / 12,672 ≈ 70.83`

current micro-circuit parameter units.

So the canonical V1 tests roughly:

> **~71 old micro-circuit-sized parameter units inside one richer cell, with only two such cells active.**

This is close to the intuitive "~100 old cells per large cell" proposal while being mathematically tied to the existing active budget rather than chosen arbitrarily.

Important: parameter matching is not FLOP matching. Bilinear elementwise operations, routing overhead, memory reads/writes, kernel efficiency, and serial execution must be measured separately.

---

# 4. Initial stored-bank size

Do not immediately create thousands of large independent cells.

Canonical first bank:

`N = 64 MacroCells`.

Independent physical cell-body parameters:

`P_bank = 64 * 897,504 = 57,440,256`.

This is intentionally moderate enough to train/test while offering substantially more role diversity than the final Top-2 active set.

Primary progression only after a successful 64-cell result:

`64 -> 128 -> 256 -> 1024`.

Do not introduce factorized virtual MacroCells in the first causal experiment. Otherwise a positive result cannot distinguish:

- better cell granularity;
- factor sharing;
- virtual capacity;
- parent-growth effects.

Factorization is a later ablation.

---

# 5. Why larger cells may help mathematically

## 5.1 Reduced routed program length

Suppose a target mapping requires `T` micro-transformations:

`F = C_T ∘ ... ∘ C_1`.

Each routing decision has probability `1-ε` of selecting a sufficiently useful unit under a simplified independent-error model. Then a rough probability of a completely correct routed path is

`P_success_micro ≈ (1-ε)^T`.

If a MacroCell compresses `g` useful local transformations and requires only

`s ≈ ceil(T/g)`

routing decisions, then

`P_success_macro ≈ (1-ε_macro)^s`.

Even if `ε_macro` is somewhat larger because each decision is coarser, reducing `T` can materially reduce cumulative routing error.

This is only an intuition model; routing errors are not actually independent. The experiment must measure real route sensitivity.

## 5.2 Better local credit assignment

With many tiny cells, a loss `L` may depend on a chain

`x_T = C_T(...C_2(C_1(x_0)))`.

The gradient to early cell parameters contains a product of Jacobians:

`∂L/∂θ_1 = ∂L/∂x_T * Π_{j=2..T} J_{C_j} * ∂C_1/∂θ_1`.

Long routed chains can increase gradient variance, attenuation/explosion, and attribution ambiguity.

A MacroCell absorbs several local transforms into one parameterized body, reducing the number of externally routed Jacobian transitions. This does not remove deep-network optimization difficulty inside the cell, but the internal path is dense/differentiable rather than discrete across many separately selected units.

## 5.3 Richer primitive class

The old micro-circuit is primarily a low-rank nonlinear map. MacroCell V1 directly represents state-memory interactions and multiplicative structure:

`W_b(P_x x ⊙ P_m r)`.

Therefore it can represent context-conditional transformations that may require several low-rank micro-circuits to approximate.

The function-class relation is

`C_micro ⊆ C_macro`

by zeroing memory/bilinear branches and choosing compatible weights.

Again: greater expressivity does not prove better generalization or trainability.

---

# 6. Why cells must not become too large

There are two bad limits.

## Too small

As `P_cell -> P_micro` and `N` grows very large:

- route search becomes harder;
- useful behavior may fragment across many units;
- per-cell gradient frequency falls;
- effective program depth rises;
- specialization may become operand/token memorization.

## Too large

As `P_cell` approaches a substantial fraction of the total model:

- each cell can become a generalist;
- routing becomes unnecessary;
- specialization entropy collapses;
- load concentrates on a few cells;
- architecture degenerates toward conventional large-expert MoE.

Therefore the project seeks a **mesoscopic optimum**:

`P_micro << P_cell << P_model`.

The V1 value `~0.9M/cell` is a test point, not a claimed universal optimum.

---

# 7. Granularity sweep — strict and limited

To avoid architecture-search swamp, only three granularity points are allowed in the first scientific sweep, all approximately active-budget matched:

### Small control

`k=8`, target `P_cell ≈ B_ref/8 ≈ 224k`.

### Canonical mesoscopic

`k=2`, `P_cell = 897,504`.

### Large control

`k=1`, target `P_cell ≈ 1.79M`.

The purpose is to test whether there is a useful intermediate granularity curve.

Do **not** test dozens of arbitrary widths/ranks before these three points are evaluated.

Expected interpretation:

- if `k=8` wins: cells may still be too coarse at ~0.9M;
- if `k=2` wins: evidence for a mesoscopic optimum;
- if `k=1` wins: routing may be less useful than expected, or tasks need very powerful single-step transforms;
- if all large-cell variants lose to the micro baseline: reject the large-cell hypothesis in its current form.

---

# 8. Routing interface

`taklif20` assumes the semantic/self-describing framework of `taklif19`, but the first causal granularity experiment should keep routing as simple and controlled as possible.

For state `s_t`:

`q_t = Q(s_t, summary(m_t))`.

For `N=64`, a full key scan is acceptable as a diagnostic baseline:

`score_i = q_t^T κ_i`.

Top-`L` candidate cells are selected, then optional candidate self-match chooses Top-`k`.

Because `N=64` is small, this first experiment should not confound MacroCell quality with a complicated hierarchical router. Hierarchical addressing becomes mandatory only when scaling beyond the diagnostic bank.

Required later property:

`C_route(N)` must be sublinear in `N`.

---

# 9. Training protocol

The biggest risk is the coupled router/cell chicken-and-egg problem. The training schedule must therefore be staged.

## Phase A — dense/soft cell pre-specialization

Use soft routing across a broader candidate set:

`p_i = softmax(score_i / τ)`

`y = Σ_i p_i y_i`.

Do not immediately use hard Top-2. This gives multiple cells gradient and allows initial functional niches to emerge.

## Phase B — specialization and role consistency

Encourage useful but not forced diversity.

Potential terms:

`L_usage` = penalty only for extreme dead-cell collapse;

`L_desc` = descriptor/body consistency from `taklif19`;

`L_route` = query/key contrastive or advantage-weighted matching.

Do not force perfectly uniform load. Natural functions may have unequal frequency.

## Phase C — sparsity annealing

A canonical schedule:

`soft-all / broad -> Top-16 -> Top-8 -> Top-4 -> Top-2`.

At each stage record quality and route utilization.

If quality collapses before Top-2 and an oracle Top-2 also fails, stop router tuning: the representation is not sufficiently sparse at this granularity.

## Phase D — hard sparse fine-tuning

Use hard Top-2 in the forward path with an appropriate straight-through, sampled, or local differentiable training mechanism.

The exact estimator is an implementation choice but must be logged because biased gradient estimators can create false conclusions.

## Phase E — end-to-end teacher transfer

For Qwen experiments, teacher objective may be

`L_total = λ_local L_FFN + λ_KL L_KL + λ_LM L_LM + λ_route L_route + λ_desc L_desc + λ_usage L_usage`.

Start with the minimum set of losses; do not turn all regularizers on simultaneously.

Recommended order:

1. `L_FFN` local functional imitation;
2. add routing;
3. hard sparsity;
4. end-to-end KL/LM;
5. add descriptor/load regularization only if audits show the corresponding failure.

---

# 10. First synthetic experiment

The old 2-step saturated benchmark is insufficient as the primary result.

Use a harder program/composition distribution with:

- variable 4/6/8-step programs;
- multiple registers;
- data-dependent next operation where possible;
- held-out operation compositions;
- held-out operand combinations;
- range/value OOD;
- deterministic full or very large evaluation sets.

Compare at matched active budget:

1. current best micro/factorized architecture;
2. ~224k-cell Top-8 control;
3. 897,504-param MacroCell Top-2;
4. ~1.79M-cell Top-1 control;
5. compact dense control with comparable active compute.

Primary claim is not raw accuracy. It is movement of the capability frontier at matched active compute.

---

# 11. First real-Qwen experiment

Teacher: cached real `Qwen3-0.6B`.

Start with late 4-6 FFN layers only.

Controls:

1. original dense Qwen FFN;
2. exact all-circuit Qwen graft;
3. old contiguous-chunk sparse student;
4. compact dense SwiGLU student;
5. learned latent basis from `taklif15` if available;
6. MacroCell Top-2 student;
7. random/matched ablations.

Train MacroCells from teacher FFN inputs/outputs, then evaluate end-to-end teacher fidelity.

Do not scale to all 28 layers until late-layer replacement passes.

Primary metrics:

- teacher-student FFN MSE/cosine;
- logit KL/MSE;
- teacher top-1 agreement;
- held-out perplexity/CE delta;
- active parameters;
- measured MACs/FLOPs proxy;
- latency and throughput;
- routing utilization;
- route reuse across semantically/functionally similar inputs.

---

# 12. Oracle tests

Before spending large effort on router optimization, measure an oracle upper bound where feasible.

Given candidate MacroCell outputs `{y_i}`, an offline oracle may search/select the subset minimizing local teacher error under `k=1/2/4`.

If even a generous oracle with trained MacroCells cannot produce acceptable quality with Top-2, the problem is representation/cell granularity, not the router.

This is a critical anti-swamp gate.

---

# 13. Failure modes and false positives

## 13.1 Giant-expert degeneration

A few MacroCells become general-purpose and receive nearly all traffic.

Symptoms:

- very low route entropy;
- top 1-2 cells dominate most inputs;
- ablation of one dominant cell destroys the model;
- semantic reuse is actually global generalism, not specialization.

Interpretation: architecture is drifting toward MoE/general dense blocks.

## 13.2 Cell cloning

Many cells learn nearly identical functions.

Symptoms:

- high output/Jacobian similarity;
- signatures differ but behavioral probes are similar;
- route permutations barely affect output.

Then stored capacity is illusory.

## 13.3 Descriptor-body mismatch

The route key says a cell belongs to one region while behavior does not match that role.

Use `taklif19` behavioral/usage signature audits.

## 13.4 Router memorizes input identity

Different operands/tokens performing the same functional role route to unrelated cells.

Required audit:

`same operation/program role + changed surface values -> high route/address reuse`.

## 13.5 MacroCell memorizes whole examples

Large cells have enough capacity to memorize data instead of reusable transformations.

OOD and held-out-program tests are mandatory.

## 13.6 Active-parameter fairness illusion

Equal active parameters do not imply equal FLOPs, memory traffic, kernel efficiency, or training cost.

Always report:

- active parameters;
- MAC/FLOP proxy;
- actual latency;
- throughput;
- physical stored parameters;
- routing overhead.

## 13.7 Serial-depth hiding

If Top-2 MacroCells internally contain many hidden recurrent steps, the comparison is no longer fair. V1 prohibits inner recurrence/attention.

## 13.8 Benchmark saturation

A 99.6% score on the already solved 2-step task is not evidence that MacroCells are better. The result must appear on harder capability-frontier/OOD tests.

## 13.9 Teacher overfit

A late-layer Qwen experiment on a tiny calibration text may show low CE delta without general fidelity. Use larger independent train/calibration/evaluation corpora before a strong claim.

## 13.10 Training-budget confound

If MacroCells receive more steps/tokens than controls, equal-compute conclusions are invalid. Use matched training budget controls where possible.

---

# 14. Measurements required for every run

Record:

```text
commit SHA
branch
seed
task/data split
training steps/tokens
optimizer/schedule
stored cell count
cell parameter count
active k
active body params
router params
router MACs
cell-body MACs
physical stored params
virtual capacity if later factorized
route entropy
cell utilization
candidate recall/oracle gap
route reuse under operand/content perturbations
quality metrics
OOD metrics
latency
throughput
peak VRAM
system RAM
failure notes
```

Never report only a virtual/model-scale label.

---

# 15. Primary GO / NO-GO gates

## Synthetic STRONG GO

At approximately the same active body budget as the current ~1.79M-active baseline, Top-2 MacroCells must show a clear advantage on harder variable-depth/OOD composition, not merely on the saturated 2-step task.

Desirable evidence:

- higher 4/6/8-step accuracy;
- better held-out-combination and value/domain OOD;
- fewer routed decisions;
- stable/reusable routes;
- no dominant-cell collapse.

## Qwen CONDITIONAL GO

On late 4-6 layers, MacroCell Top-2 should materially improve the quality/active-compute frontier over contiguous Qwen chunk sparsity and should be competitive with or better than a compact dense student at comparable compute.

A particularly strong signal would be near-teacher fidelity at roughly the current active budget with only two active cells.

## NO-GO

Stop the large-cell direction in its present form if, after matched-budget training and the limited granularity sweep:

- Top-2 MacroCells do not outperform micro-circuit and compact-dense controls on difficult/OOD tasks;
- oracle Top-2 remains poor;
- cells collapse into a few generalists;
- knowledge still requires a large fraction of cells active;
- routing/dispatch overhead removes any systems advantage;
- gains disappear on independent evaluation data.

Do not respond to NO-GO by testing dozens of arbitrary cell widths.

---

# 16. Scaling only after GO

If the 64-cell independent bank passes:

### Step 1

`64 -> 128 -> 256` independent MacroCells while keeping Top-2 active.

Test whether stored capacity improves capability without raising active compute.

### Step 2

Introduce hierarchical semantic addressing from `taklif19` when full scanning becomes inappropriate.

### Step 3

Test factorized virtual MacroCells:

`shared bases + small per-cell codes -> many virtual MacroCells`.

This connects to the successful factorized virtual micro-circuit line, but it must be a separate ablation.

### Step 4

Test parent-growth:

`64 trained cells -> larger bank`

while preserving learned cells/addresses and adding new capacity.

Use equal-cumulative-training controls.

### Step 5

Only after the above, integrate with `taklif16` Transformer -> NE cross-architecture transplant.

---

# 17. Scientific interpretation

A positive result would not prove that "large neurons are universally superior". It would support a narrower and more useful statement:

> For Neural Engine-style sparse dynamic computation, there exists a coarser learned computational granularity at which the same active parameter budget can express and learn useful multi-step transformations with fewer routing decisions and better capability/generalization than very small routed circuits.

A negative result is equally informative: it would imply that the main bottleneck is not simply cell granularity and would strengthen the case for learned latent bases, representation learning, or other architectures rather than endlessly increasing cell size.

---

# 18. Canonical V1 configuration

The first serious test is frozen as:

```text
state dimension d       = 384
memory/read dimension   = 384
MacroCell hidden h      = 480
bilinear rank b         = 128
cell body params        = 897,504
stored MacroCells N     = 64
candidate count L       = 8
active MacroCells k     = 2
active body params      = 1,795,008
reference active params = ~1,792,305
active-budget delta     = ~+0.151%
execution               = serial residual read-compute-write
inner attention         = none
inner recurrence        = none
inner router            = none
```

This configuration must be tested before architecture complexity is increased.

---

# 19. Final rule

The purpose of `taklif20` is to answer one question cleanly:

> **With essentially the same active parameter budget, does Neural Engine benefit more from many tiny routed primitives or from only two mesoscopic, stateful computational cells that each contain roughly seventy old micro-circuit parameter units?**

Until that question is answered, do not claim that bigger cells solve the Neural Engine representation problem, and do not expand the design into an open-ended collection of large-cell variants.
