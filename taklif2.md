# `taklif2.md` — Neural Engine V2 reasoning-core proposal

## Purpose

This file is a design and falsification handoff for the next Neural Engine architecture iteration.

The current evidence says the systems side of Neural Engine is increasingly convincing, while the reasoning core is now the dominant bottleneck. Small patches such as coverage loss, larger active-circuit budgets, route exploration, longer curriculum, and parent-based capacity growth help locally, but they have not produced a large capability jump or a reliable 300M -> 500M compositional scaling law.

The recommendation is therefore **not** to discard the project and **not** to turn it into a Transformer. The recommendation is to freeze the current architecture as **NE-V1**, preserve all systems work, and build a materially different **NE-V2 reasoning core** around the same sparse-capacity philosophy.

The target philosophy remains:

```text
Huge stored capacity
        ↓
cheap addressing / routing
        ↓
small relevant working set
        ↓
dynamic reusable computation
        ↓
persistent state / memory
        ↓
more steps only when needed
```

The key change is:

```text
V1: tiny subset = mostly parallel weighted circuit blocks
V2: tiny subset = a compositional micro-program over structured working memory
```

---

# 1. Why a larger architecture change is now justified

The current experimental pattern is consistent enough that another sequence of local patches is unlikely to be the highest-value next step.

Important observations already established in the repository:

- Sparse/fixed-active execution survives large stored capacity.
- The 300M and 500M models are hardware-feasible on an RTX 3060-class GPU.
- `LazyAdamW` can preserve a useful learning trajectory while updating routed rows instead of treating the entire bank as equally active.
- CPU-RAM circuit paging is feasible and caching materially reduces transfer traffic.
- Parent-based growth is much better than cold random bank expansion on the easier numeric benchmark.
- Route exploration helps credit assignment in some composition runs.
- Curriculum improves the 300M composition result.
- Coverage improves utilization but does not automatically improve quality.
- Increasing active circuits from 8 to 16 does not create a reliable capability jump.
- Parent growth does not reliably turn 500M into a better compositional model than 300M.
- The audited deterministic composition benchmark currently shows roughly:
  - 20M reference: about 12.40% hidden-pair accuracy.
  - 300M curriculum mean: about 13.79%.
  - 500M curriculum-growth mean: about 12.77%.

The conclusion is not that sparse capacity is useless. The conclusion is that **the current mechanism does not reliably convert additional stored circuits into deeper compositional computation**.

This is now an architecture problem, not merely a hyperparameter problem.

---

# 2. Preserve what already works

Do **not** rewrite everything from scratch.

The following parts are valuable and should remain available in NE-V2 unless an experiment directly falsifies them.

## 2.1 Sparse circuit-bank philosophy

Keep the separation between:

- large total/stored capacity;
- small active computation per example/step.

This is one of the strongest results of the project so far.

## 2.2 Hierarchical routing as a cheap addressing concept

Do not revert to scoring every circuit densely.

The exact routing geometry should change, but the principle that routing cost must remain sublinear in total bank size should remain.

## 2.3 Lazy/sparse optimizer work

Preserve `LazyAdamW` and continue validating it.

The optimizer is orthogonal to the reasoning-core redesign and may become even more valuable if V2 has more structured circuit families and locality.

## 2.4 CPU-RAM / GPU-cache direction

Preserve the memory-placement work.

The architecture should continue to be designed so a large cold bank can live in system RAM while only a hot working set lives on the accelerator.

## 2.5 Parent -> child capacity growth

Preserve the ability to expand a trained model by cloning/splitting useful circuits rather than creating a completely cold larger bank.

Growth should become more targeted in V2 rather than being discarded.

## 2.6 Adaptive recurrent execution philosophy

Keep the idea that hard tasks can consume more internal steps than easy tasks.

However, future large-language versions must eventually move away from externally supplied depth targets and toward autonomous halting/confidence/ponder mechanisms.

---

# 3. Freeze NE-V1 before major changes

Create a clear V1 freeze point.

V1 should retain:

- the current benchmarks;
- checkpoints;
- trained 20M/100M/300M/500M results;
- LazyAdamW implementation;
- RAM paging prototype;
- capacity-growth tooling;
- deterministic composition evaluator;
- all negative results.

Do not silently replace V1 behavior with V2 behavior and then compare across stale results.

Recommended conceptual split:

```text
NE-V1
  = current single-state + routed low-rank bank architecture

NE-V2
  = structured working memory + compositional micro-program architecture
```

V1 remains the control.

---

# 4. Main V2 change: replace the single reasoning vector with multi-slot working memory

The current architecture compresses too many logically different things into one 384-dimensional state:

- task identity/context;
- operands;
- intermediate values;
- control state;
- routing history;
- partial results;
- next-step intent.

For shallow tasks this can work. For composition it may create destructive interference.

NE-V2 should use **multiple persistent working-memory slots**.

Initial prototype suggestion:

```text
8 slots × 256 or 384 dimensions
```

Example semantic roles, learned or partially scaffolded:

```text
slot 0: global task/context
slot 1: operand / entity A
slot 2: operand / entity B
slot 3: intermediate result
slot 4: current operator / control context
slot 5: scratch
slot 6: scratch
slot 7: persistent summary / program state
```

These labels do not need to be permanently hand-coded. The point is to provide **separable writable memory locations**.

The router/circuit executor should be able to choose:

```text
read slots: 1, 2, 4
write slot: 3
```

and later:

```text
read slots: 3, 5
write slot: 6
```

rather than forcing every transformation through a single monolithic vector.

## 4.1 Sparse slot updates

Do not update every slot at every step.

A micro-operation should explicitly select:

- read slot(s);
- circuit/family;
- write slot(s);
- optional residual/preserve behavior.

This preserves sparsity not only across parameters but also across working memory.

## 4.2 Slot protection / write gating

The previous generic gated-memory experiment was too expensive and did not become a default win.

V2 should not simply add a dense gate over the whole state.

Instead, write operations should be **structural and local**:

```text
selected slots change
unselected slots persist exactly or nearly exactly
```

This makes intermediate reasoning easier to preserve.

## 4.3 Read/write addressing

The model should learn both:

- which circuit/family to execute;
- which memory slots to read/write.

This can be factored into small discrete decisions so routing cost stays cheap.

---

# 5. Replace mostly-parallel circuit mixing with explicit compositional micro-programs

The current parallel circuit mode often behaves roughly like:

```text
C1 ┐
C2 ├─ weighted combination -> delta
C3 ┘
```

This is useful for conditional feature transformation, but it may be structurally weak for ordered reasoning such as:

```text
(a + b) * c
```

where the desired computation is closer to:

```text
ADD(a, b)
   ↓
store intermediate
   ↓
MULTIPLY(intermediate, c)
```

NE-V2 should make **ordered circuit composition a first-class operation**.

## 5.1 Do not repeat the old naive serial experiment

The old serial-circuit experiment simply applied selected blocks sequentially and was slower without enough quality gain.

V2 should instead let the router construct a **small explicit micro-program**.

The important difference is:

```text
old serial:
select K blocks -> apply all in fixed sequence

V2 micro-program:
select computation topology + read/write slots + node-specific routes
```

## 5.2 Limit topology for hardware efficiency

Do not start with arbitrary dynamic graphs.

Use a small set of pre-defined templates that can later be fused into efficient kernels.

Initial templates:

```text
1. single node
   A

2. chain-2
   A -> B

3. chain-3
   A -> B -> C

4. parallel-2 + merge
   A --\
        M
   B --/

5. fork-join
        B
   A ->   -> D
        C
```

The first prototype may only need **single-node + chain-2**.

If chain-2 materially improves the composition benchmark, expand later.

## 5.3 Program length vs recurrent steps

Separate two concepts:

- micro-program nodes within one reasoning step;
- recurrent reasoning steps over persistent memory.

A hard problem may use both:

```text
step 1: A -> B
state/memory update
step 2: C
state/memory update
step 3: D -> E
```

This is more expressive than merely increasing active circuits in parallel.

---

# 6. Redesign routing around semantic / learned circuit families

The current router ultimately maps to a local contiguous candidate window around an address. At large scale, index proximity has no guaranteed semantic meaning.

This becomes increasingly suspicious as the bank grows.

V2 should add a **family level** above the individual circuit level.

Conceptually:

```text
state / memory
      ↓
family router
      ↓
selected family or small family set
      ↓
local circuit router inside family
      ↓
selected circuit(s)
```

Possible learned family examples:

```text
Arithmetic
  ├─ add-like
  ├─ multiply-like
  ├─ modular
  └─ comparison

Memory
  ├─ store
  ├─ retrieve
  └─ match

Control
  ├─ branch
  ├─ continue
  └─ halt
```

These are examples, not mandatory hard-coded labels.

## 6.1 Learned family membership

Several approaches are acceptable to test:

- fixed-size learned buckets;
- learned centroid/cluster keys;
- product-key style multi-address families;
- parent-child growth families;
- semantic locality learned from co-activation / gradient statistics.

The key requirement is that nearby candidates should have **meaningful learned locality**, not arbitrary row-number locality.

## 6.2 Multiple addresses

Consider allowing the router to request two or more independent family addresses when composition requires distinct primitives.

Example:

```text
address 1 -> arithmetic primitive family
address 2 -> memory/control family
```

This may be better than forcing one local window to contain every useful computation type.

---

# 7. Improve credit assignment: soft/local during training, hard during inference

The route-exploration experiments strongly suggest that the current hard router can lock into suboptimal regions too early.

Inference should remain sparse and hard.

Training should receive better gradient/exploration support.

Recommended direction:

```text
TRAINING
state
 ↓
cheap family routing
 ↓
small candidate set (for example 16–64)
 ↓
soft / differentiable local competition
 + controlled exploration
 ↓
selected sparse program

INFERENCE
state
 ↓
hard family + hard local route
```

Do **not** evaluate all circuits softly.

Softness should only exist inside a small candidate neighborhood/family.

Possible mechanisms to test:

- straight-through top-k;
- Gumbel-softmax / Gumbel-top-k;
- sparsemax / entmax over a local pool;
- sampled candidate alternatives;
- stochastic branch exploration with annealing;
- epsilon-greedy exploration that decays over training;
- uncertainty-guided exploration;
- underused-family exploration.

## 7.1 Exploration should eventually become targeted

The existing fixed 5% random branch exploration is a useful probe, but likely not the final mechanism.

Better target:

```text
if route confidence is low -> explore more
if family is under-trained -> explore more
if route is stable/high-confidence -> explore less
```

This keeps noise where it is useful.

---

# 8. Make individual circuits somewhat stronger

The current low-rank circuit primitive is approximately:

```text
384 -> rank 16 -> 384
```

This is extremely cheap, which is good for systems efficiency, but it may be too weak to become a stable meaningful primitive.

V2 should test stronger circuit primitives while preserving a small active set.

Suggested sweep:

```text
rank 16
rank 32
rank 64
```

and potentially a slot-aware block:

```text
selected read slots
      ↓
projection / combine
      ↓
nonlinear low-rank block
      ↓
selected write slot
```

The objective is not “make every circuit huge.”

The objective is to determine whether fewer, stronger, more semantically stable primitives compose better than tens of thousands of very weak rows.

For example, a 500M model may be more useful as:

```text
10k–20k stronger circuits
```

than:

```text
38k+ ultra-small circuits
```

if active compute remains small and composition improves.

---

# 9. Capacity growth should become specialization-aware

Keep the parent-based growth result, but make it more targeted.

Do not blindly clone the hottest circuits.

Instead try to identify **overloaded circuits** — circuits that appear to serve multiple incompatible contexts or tasks.

Signals for overload can include:

- high selection count across heterogeneous task types;
- high gradient variance by context;
- conflicting gradient directions;
- high output variance conditioned on route context;
- large loss sensitivity when replaying/swapping its route;
- high co-activation with different families;
- route entropy around the circuit.

Then split:

```text
parent C
  ↓
C1 = C + small perturbation
C2 = C + small perturbation
C3 = C + small perturbation
```

and encourage specialization by context.

This is closer to developmental capacity growth:

```text
useful computation
      ↓
overload detected
      ↓
split
      ↓
specialize
      ↓
new useful capacity
```

rather than:

```text
add random dormant rows
```

---

# 10. Working-memory and circuit-composition experiments should happen at 20M first

Do **not** start NE-V2 at 300M or 500M.

The current audited hidden-pair composition benchmark gives a cheap falsification target.

Approximate V1 reference:

```text
20M hidden-pair deterministic accuracy ≈ 12.4%
```

Initial V2 gate:

```text
FAIL:
  <= ~14–15%

WEAK SIGNAL:
  ~16–20%

STRONG SIGNAL:
  >20%

VERY STRONG SIGNAL:
  25–30%+
```

These thresholds are not sacred, but the point is that V2 must create a **large architecture-level jump**, not another +0.5 or +1 point patch.

If 20M V2 does not clearly beat V1, do not scale it.

---

# 11. Recommended minimal V2 prototype

Do not implement every idea in this file at once.

Start with a minimal architecture that tests the central hypothesis.

## V2-minimum

```text
1. 8 persistent memory slots
2. slot-selective read/write
3. family router
4. local circuit router
5. chain-2 micro-program template
6. hard sparse inference
7. local differentiable/stochastic training route
8. existing LazyAdamW compatibility
```

Keep total parameters around ~20M for the first run.

Keep active compute explicitly measured.

Do not hide a 10x active-compute increase behind a quality gain.

---

# 12. Required ablation sequence

The goal is to know **which change caused the improvement**.

Recommended sequence:

```text
A. V1 control

B. V1 + multi-slot memory only

C. B + slot-selective writes

D. C + chain-2 micro-program

E. D + family routing

F. E + local differentiable / stochastic training routing

G. rank 16 vs 32 vs 64 on best architecture
```

If time is limited, combine B/C and then test D separately.

The most important falsification is whether **explicit ordered composition** produces a large gain.

---

# 13. Benchmark requirements for V2

Do not rely only on the existing easier numeric benchmark.

Use at least:

## 13.1 Hidden operation-order composition

Primary architecture gate.

Use the deterministic full grid evaluator, not tiny random samples.

## 13.2 Longer composition depth

After depth-2 starts working, add:

```text
(a op1 b) op2 c
```

then:

```text
((a op1 b) op2 c) op3 d
```

and potentially mixed control/memory tasks.

## 13.3 Intermediate-state probes

Inspect whether slots specialize.

Examples:

- does one slot consistently contain intermediate arithmetic results?
- do different operations route to distinguishable families?
- does the second node of a chain causally depend on the first node output?

## 13.4 Route/program replay

Perform causal tests analogous to earlier route-swap tests:

- replace node 1 route only;
- replace node 2 route only;
- swap write slot;
- replay the same program on changed operands;
- replace family but preserve local circuit index;
- perturb memory slot content.

A genuine compositional system should show structured causal sensitivity.

---

# 14. Scaling gates after V2 succeeds at 20M

Only scale if the 20M result shows a clear architecture-level win.

Suggested progression:

```text
20M
 ↓
50–100M
 ↓
300M
 ↓
500M
 ↓
1B / 1.5B systems + quality test
```

A desirable pattern would be something like:

```text
20M  = clearly above V1
100M = clearly above 20M
300M = clearly above 100M
500M = clearly above 300M
```

The exact percentages will depend on the benchmark, but the ordering matters.

If quality plateaus while active compute stays fixed, investigate whether the active program budget itself must grow mildly with capacity.

Possible controlled scaling:

```text
20M: 1–2 program nodes
100M: 2 nodes
300M: 2–3 nodes
500M+: 2–4 nodes
```

The design goal should be **sublinear active-compute growth**, not necessarily mathematically constant active compute forever.

---

# 15. Do not insist on perfectly constant active compute if it blocks capability

The original fixed-active result is valuable, but it should not become a dogma.

A model with:

```text
20M total -> 2M active
500M total -> 4M active
7B total -> 50M active
```

can still be revolutionary compared with dense execution.

The correct objective is:

```text
capability grows much faster than active compute
```

not:

```text
active compute must never grow at all
```

This is especially important for reasoning depth.

---

# 16. Separate systems scaling from capability scaling

Continue to report two independent tracks.

## Systems track

Measures:

- total parameters;
- active parameters;
- active fraction;
- forward/backward throughput;
- optimizer cost;
- VRAM;
- CPU RAM;
- H2D bytes;
- cache hit rate;
- GPU idle time;
- prefetch overlap;
- page reuse;
- active program length.

## Capability track

Measures:

- deterministic hidden-pair composition;
- train/held-out gap;
- depth-2/depth-3/deeper tasks;
- true held-out structures;
- program/route causal dependence;
- quality vs stored capacity;
- quality vs active compute.

A 1.5B model can be a successful **systems milestone** even if it is not yet a successful **quality milestone**.

Do not mix those claims.

---

# 17. 1B / 1.5B recommendation

A 1B–1.5B RAM-backed run is still useful for systems validation, but it should not be presented as capability scaling until V2 solves the smaller composition gate.

Useful 1.5B systems test sequence:

```text
200 steps
 ↓
1000 steps
 ↓
5000 steps
```

Measure:

- system RAM usage;
- optimizer-state RAM;
- active working-set size;
- GPU cache size;
- cache hit rate;
- H2D bandwidth;
- writeback bandwidth;
- forward time;
- backward time;
- optimizer time;
- GPU idle fraction;
- loss trajectory;
- active-program statistics.

Do this independently of the V2 quality gate if desired.

However, do **not** assume that 1.5B will become smarter simply because it contains more stored rows.

---

# 18. CPU-RAM training architecture target

Long-term memory placement should look approximately like:

```text
CPU RAM
├─ cold circuit/family weights
├─ LazyAdamW moments
├─ inactive circuit rows
├─ growth metadata
└─ optional route statistics

GPU / accelerator
├─ router/controller
├─ working-memory slots
├─ hot circuit-family cache
├─ active micro-program nodes
├─ current activations
└─ current gradients
```

Per reasoning step:

```text
working memory
   ↓
family route
   ↓
local candidates
   ↓
cache hit?
 ┌───────┴────────┐
yes              no
 ↓                ↓
compute      prefetch/page
                  ↓
              compute
   ↓
slot-selective writes
   ↓
next reasoning step
```

Training must preserve the full sample trajectory and backward dependencies across the selected micro-program.

Do not group or page circuits in a way that destroys program semantics.

---

# 19. Hardware/locality design matters from the start

V2 should not become an elegant research graph that is impossible to execute efficiently.

Design every dynamic decision with batching/cache locality in mind.

Prefer:

- fixed topology templates;
- contiguous family pages;
- batched slot reads/writes;
- batched circuit-row fetches;
- predictable family-level memory regions;
- prefetchable candidate groups;
- limited program lengths;
- fused kernels where possible.

Avoid:

- arbitrary per-scalar memory gathers;
- arbitrary dynamic graph sizes;
- thousands of tiny independent kernel launches;
- full-bank soft routing;
- per-sample Python loops in production paths.

---

# 20. V2 should remain clearly distinct from conventional MoE

Do not drift into a standard top-k expert architecture unless the evidence forces that conclusion.

The intended V2 computation is:

```text
persistent multi-slot state
      ↓
route program node 1
      ↓
execute reusable primitive
      ↓
write selected memory slots
      ↓
route program node 2 based on updated state
      ↓
execute another primitive
      ↓
repeat if necessary
```

This differs from conventional MoE where a token often selects a small number of large feed-forward experts inside a mostly fixed layer stack.

The important Neural Engine property is **dynamic compositional trajectory**, not merely sparse parameter selection.

---

# 21. Failure criteria

NE-V2 should be falsifiable.

Treat these as warning signs:

## FAIL A — no composition jump

If 20M V2 stays near ~12–15% on the audited hidden-pair benchmark, the redesign did not solve the core issue.

## FAIL B — quality gain comes only from huge active compute

If quality rises only because V2 activates 5–10x more parameters, the architecture may be losing its central advantage.

## FAIL C — memory slots do not specialize or matter causally

If slot permutations/writes barely affect outputs, the structured memory may be decorative.

## FAIL D — micro-program order does not matter

If swapping A -> B into B -> A barely changes behavior on order-sensitive tasks, the program representation is not genuinely compositional.

## FAIL E — larger bank still fails to beat smaller bank

After V2 works at 20M, if 100M/300M still do not produce reliable improvements, capacity conversion remains unsolved.

## FAIL F — routing/training instability

If local differentiable routing causes collapse, exploding memory traffic, or loss of sparse inference semantics, redesign the training router rather than hiding the problem.

---

# 22. Positive evidence criteria

Strong evidence for V2 would look like:

- 20M V2 materially exceeds 20M V1 on deterministic composition.
- The gain persists across at least 3 seeds.
- Program-node order is causally important.
- Memory-slot writes are causally important.
- Active parameters remain far below total parameters.
- 100M/300M improve quality over 20M with only mild active-compute growth.
- Parent-based split/growth creates specialization rather than dormant rows.
- RAM/GPU paging still works with family locality.
- LazyAdamW remains numerically viable.

A very strong result would be a monotonic curve like:

```text
20M  < 100M < 300M < 500M
```

on a benchmark that truly requires held-out composition rather than task-ID memorization.

---

# 23. Suggested experiment report format

Every V2 experiment should record:

```text
Experiment name
Architecture variant
Seed
Total params
Active params
Active fraction
Memory slot count
Slot dimension
Circuit rank
Program topology
Average program nodes
Average recurrent steps
Router candidate pool
Exploration method
Training steps
Training samples
Train accuracy
Deterministic held-out accuracy
Depth breakdown
Dead circuits/families
Family utilization
Route entropy
Program entropy
Peak VRAM
CPU RAM
H2D bytes
Cache hit rate
Train throughput
Inference throughput
Result classification:
  FAIL / WEAK SIGNAL / STRONG SIGNAL
Interpretation
Next falsification
```

Do not report only the best seed.

Preserve negative results.

---

# 24. Immediate implementation plan

## Phase 0 — freeze and branch

- Freeze current NE-V1.
- Create NE-V2 architecture path without deleting V1.
- Keep current datasets/evaluator unchanged as controls.

## Phase 1 — structured memory

Implement:

- 8 persistent slots;
- slot-selective read/write;
- sparse slot update;
- basic instrumentation.

Test at ~20M.

## Phase 2 — chain-2 micro-program

Implement:

- one-node and two-node program templates;
- node 2 routing conditioned on memory after node 1;
- causal route/program instrumentation.

Test the hidden-pair composition benchmark.

## Phase 3 — family routing

Implement:

- family keys/buckets;
- local candidate routing inside family;
- family utilization metrics;
- cache-friendly family layout.

## Phase 4 — training credit assignment

Compare:

- hard routing;
- 5% random exploration;
- annealed exploration;
- local differentiable routing.

## Phase 5 — circuit strength sweep

Compare rank 16/32/64 or equivalent stronger primitive.

## Phase 6 — scale only the winner

Run:

```text
20M -> 100M -> 300M
```

Only continue to 500M if quality improves.

## Phase 7 — combine with growth

Use overload-aware parent splitting instead of blind hottest-row cloning.

## Phase 8 — 1B/1.5B systems test

Use RAM-backed sparse training independently to validate systems scale.

---

# 25. One recommended concrete V2 architecture

A practical first implementation could be:

```text
Input encoder
   ↓
8 × 256-d persistent memory slots
   ↓
small global control vector
   ↓
family router
   ↓
local candidate router
   ↓
program template selector
   ↓
chain length 1 or 2
   ↓
for each node:
   choose read slots
   choose circuit
   execute rank-32 slot-aware micro-block
   choose write slot
   update only selected slot
   ↓
recurrent step controller
   ↓
optional next micro-program
   ↓
output head
```

Initial constraints:

```text
memory slots: 8
slot dim: 256 or 384
families: 32–128 learned families
candidate circuits per family: small local pool
active circuits per node: 1–4
program nodes: 1–2 initially
recurrent steps: <=3 initially
circuit rank: 32 first candidate
```

The exact numbers should be ablated; the structural idea is the important part.

---

# 26. What not to do next

Do not spend the next major budget on:

- a blind 700M/1B quality run with the current V1 core;
- another coverage-only tweak;
- simply increasing `active_circuits`;
- simply increasing recurrent steps without better memory structure;
- another random larger circuit bank;
- another result based only on tiny random evaluation samples;
- claiming capability scaling from easier numeric tasks alone.

Those experiments may still be useful as controls, but they are no longer the main research direction.

---

# 27. Strategic interpretation

The strongest current story is:

```text
Systems architecture: promising
Sparse training: promising
RAM paging: promising
Fixed-active capacity: promising
Reasoning/capability scaling: bottleneck
```

This is a good research position because the bottleneck is now narrow enough to attack directly.

The project does not need another superficial patch. It needs a reasoning machine that can turn sparse stored primitives into an ordered computation.

The central V2 hypothesis should be:

> **A large sparse neural system will convert stored capacity into reasoning capability more effectively when it composes reusable circuit primitives as explicit short programs over persistent structured working memory, rather than mixing many selected blocks into one monolithic recurrent state.**

Everything in NE-V2 should be designed to falsify or support that hypothesis.

---

# 28. Final recommendation to the agent

Read this file together with `taklif.md` and the latest deterministic composition reports.

Do not treat every suggestion here as something that must be implemented simultaneously.

Use this order of priority:

```text
1. Freeze V1.
2. Build multi-slot persistent working memory.
3. Add true chain-2 compositional execution.
4. Add meaningful family/local routing.
5. Improve training credit assignment locally.
6. Prove a large 20M composition gain.
7. Scale only after the gain is repeatable.
8. Keep LazyAdamW/RAM systems work intact throughout.
```

The immediate success criterion is not a larger parameter count.

The immediate success criterion is:

> **a much stronger compositional reasoning result at roughly the same scale and active compute.**

If that succeeds, then 100M/300M/500M/1B scaling becomes scientifically meaningful again.
