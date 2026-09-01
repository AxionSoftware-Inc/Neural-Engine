# Neural Engine

Experimental research repository for a **non-Transformer, non-MoE neural architecture** designed around one central hypothesis:

> **Model capacity should not require proportional computation per inference step.**
>
> A large model should be able to keep a large amount of learned capacity while activating and touching only a small, input-dependent subset of parameters for the current computation.

This repository starts from first principles. The initial goal is **not** to build a production LLM, not to generate fluent text, and not to beat state-of-the-art models. The first goal is to determine whether a new architecture can achieve useful learned behavior while using dramatically fewer active parameters and less memory traffic than a comparable dense Transformer.

---

## 1. Research question

Dense Transformer inference couples total model size to per-token weight traffic. If a dense model contains `N` parameters, a large fraction of those parameters are touched repeatedly during inference.

We want to test whether we can instead build a system with:

- large total learned capacity;
- a small persistent controller/state;
- cheap hierarchical routing;
- many small reusable computational circuits;
- input-dependent circuit activation;
- adaptive computation depth;
- structured memory access;
- no self-attention requirement;
- no Transformer blocks;
- no conventional top-k expert MoE design.

The key scaling target is:

```text
Total parameters   grows strongly
Active parameters  grows slowly or remains nearly constant
```

If this relationship can be demonstrated at small scale without destroying task quality, the architecture is worth scaling and optimizing.

---

## 2. Hardware constraint

Primary development/training machine:

```text
GPU: NVIDIA RTX 3060
VRAM: 12 GB
```

Design all initial experiments so they comfortably fit this GPU.

The architecture must not depend on multi-GPU training.

The first iteration should favor **fast experimental turnaround** over model size.

Recommended progression:

```text
V0:  20-30M total parameters
V1:  40-50M total parameters
V2:  ~100M total parameters
```

Do not start with a 100M model unless smaller experiments have already shown a positive signal.

---

## 3. What this project is NOT

The following are specifically not the intended architecture:

- a standard Transformer with pruning;
- a Transformer with sparse attention;
- a conventional MoE with a few large experts;
- a wrapper around an existing LLM;
- retrieval-augmented generation as a substitute for model computation;
- a language-model benchmark project;
- a CUDA optimization project before the learning principle is validated.

Existing ideas may be used as references, but the core forward computation should remain independently designed.

---

## 4. Core hypothesis

Assume a model contains many learned computational micro-circuits.

For a given input/state, only a small subset should be necessary.

Instead of evaluating every block, the model should:

```text
input
  |
  v
small encoder
  |
  v
persistent state
  |
  v
cheap hierarchical address/router
  |
  v
small candidate set
  |
  v
selected micro-circuits only
  |
  v
state update
  |
  +----> repeat if needed
  |
  v
output
```

Conceptually:

```text
C_t = Router(x, s_t)

s_(t+1) = Compute(C_t, x, s_t; theta[C_t])
```

where:

- `theta` is the full parameter set;
- `C_t` is a small active subset/circuit set;
- `|theta[C_t]| << |theta|`;
- routing itself must be much cheaper than evaluating the whole model.

A crucial requirement is that the router must **not score every stored circuit with a large dense operation**. That would merely move the dense bottleneck into routing.

---

## 5. V0 target architecture

V0 should be deliberately simple.

Suggested total size:

```text
~25-30M parameters total
```

Suggested rough allocation:

```text
Input encoder / embeddings       1-3M
Persistent controller/state      2-4M
Hierarchical routing             <1-2M
Sparse circuit bank              18-24M
Output head                      1-2M
```

Target active parameters per reasoning step:

```text
~3-6M
```

The exact numbers are not mandatory. The important requirement is that **active compute is several times smaller than total capacity**.

### 5.1 Persistent state

Use a small learned recurrent state rather than a Transformer context stack.

Possible initial state size:

```text
256-1024 dimensions
```

The state should carry information across multiple internal computation steps.

V0 may use a GRU-like or custom gated update if needed for stability. The use of a recurrent mathematical primitive is allowed; the architecture itself must still remain non-Transformer.

### 5.2 Circuit bank

Do not use a few large experts.

Use many small reusable parameter blocks.

Example:

```text
1024-4096 micro-circuits
```

Each circuit should be small enough that many combinations can represent different computation paths.

Preferred property:

```text
same circuit may participate in many tasks
```

rather than:

```text
one expert == one whole domain
```

The circuit granularity should also remain hardware-friendly. Avoid individual-neuron random gathers in the first implementation. Use contiguous blocks/tensors so selected circuits can be batched efficiently.

### 5.3 Hierarchical routing

The router is the most important architectural component.

Do not perform a single dense score over all circuits if avoidable.

V0 should test a hierarchical routing tree.

Example:

```text
state
  |
  v
level 1: choose among 8 groups
  |
  v
level 2: choose among 8 subgroups
  |
  v
level 3: choose among 8 leaves
  |
  v
small local candidate pool
  |
  v
select k circuits
```

With branching factor 8 and depth 4, thousands of locations can be addressed through a small number of decisions.

Alternative routing approaches may later include:

- learned hashes;
- product keys;
- tree routing;
- locality-sensitive addressing;
- approximate nearest-neighbor routing;
- learned discrete addresses;
- routing from multiple state components.

But V0 should remain easy to debug.

### 5.4 Structured sparse execution

The active set should be composed of **blocks**, not isolated random weights.

This is essential because real CPU/GPU/NPU hardware benefits from contiguous memory and matrix operations.

The long-term principle is:

> semantic sparsity + structured physical sparsity

### 5.5 Adaptive internal depth

A fixed number of internal steps is acceptable for the first implementation.

However, design the interface so later versions can support:

```text
simple input  -> few internal steps
hard input    -> more internal steps
```

A learned halt/continue decision may be added only after the basic architecture trains reliably.

---

## 6. Initial task domain

Do **not** begin with natural-language generation.

We need controllable tasks where correctness is exact and unlimited training examples can be generated for free.

Create a synthetic multi-skill benchmark.

Recommended task families:

1. integer arithmetic;
2. comparisons;
3. boolean logic;
4. sequence transformations;
5. sorting small sequences;
6. key-value lookup;
7. associative recall;
8. graph traversal;
9. multi-hop relation following;
10. conditional computation;
11. composition of two or more operations;
12. simple algorithmic state machines.

Examples:

```text
INPUT:
A = 17
B = 8
RULE = if A > B then (A-B)*3+2 else A+B
OUTPUT:
29
```

```text
MEMORY:
Alice -> red -> 17
Bob   -> blue -> 28
John  -> green -> 43
QUERY:
Bob.value + Alice.value
OUTPUT:
45
```

```text
EDGES:
A -> B
B -> D
D -> F
C -> E
QUERY:
node three hops after A
OUTPUT:
F
```

Training data should be procedurally generated so train/validation/test sets can contain different values and graph instances.

Prevent memorization by using fresh random instances for held-out evaluation.

---

## 7. Why synthetic tasks first

The first experiment is about architecture economics, not language competence.

Synthetic tasks provide:

- exact ground truth;
- unlimited data;
- cheap generation;
- controlled difficulty;
- explicit task composition;
- direct measurement of generalization;
- the ability to inspect whether relevant circuits activate;
- very short training cycles.

Language modeling should be attempted only after the sparse architecture demonstrates a clear signal.

---

## 8. Dense baseline

Every claimed result must be compared against a dense baseline.

Implement a small Transformer baseline using the **same input/output representation and same dataset**.

Baseline target:

```text
~20-30M parameters
```

Match total parameter count as closely as practical.

Do not intentionally handicap the Transformer.

Use a standard, competent implementation with:

- pre-norm blocks;
- causal or task-appropriate attention;
- GELU/SwiGLU;
- AdamW;
- mixed precision where stable.

The baseline is the control experiment.

---

## 9. Required metrics

Never report only accuracy.

For every experiment record at least:

### Quality

- train loss;
- validation loss;
- exact-match accuracy;
- accuracy by task family;
- accuracy by reasoning depth;
- out-of-distribution/generalization accuracy.

### Architecture

- total parameter count;
- active parameter count per step;
- active parameter fraction;
- number of active circuits;
- routing decisions per step;
- average internal steps.

### Compute

- estimated FLOPs/sample;
- estimated active weight bytes/sample;
- measured samples/sec;
- forward latency;
- training step latency;
- peak VRAM;
- GPU utilization if practical.

### Routing

- circuit utilization distribution;
- router entropy;
- percentage of dead circuits;
- percentage of always-hot circuits;
- routing stability across related inputs;
- overlap of active circuits between task families.

---

## 10. Important distinction: algorithmic vs hardware speed

A sparse architecture may initially touch 5x fewer parameters but run slower in PyTorch because dense matrix multiplication is extremely optimized while sparse gather/scatter and many small kernels are not.

This must **not** be confused with architectural failure.

Always separate:

```text
A. algorithmic efficiency
   - active params
   - bytes touched
   - FLOPs

B. current implementation efficiency
   - wall-clock latency
   - GPU utilization
```

If A is strongly positive and B is negative, a later custom CUDA/C++ kernel may be justified.

If A is not positive, kernel optimization is premature.

---

## 11. Primary acceptance test

The project should answer this question first:

> Can a model with comparable total capacity to a dense Transformer achieve comparable useful task performance while activating substantially fewer parameters per computation?

### Strong positive signal

Any result close to the following is highly interesting:

```text
Dense Transformer:
  total params  ~30M
  active params ~30M
  accuracy      95%

Neural Engine:
  total params  ~30M
  active params <=5-8M
  accuracy      ~90-95%+
```

Exact thresholds may vary, but we want a clear Pareto improvement.

### Weak positive signal

Worth continuing if:

- quality is somewhat below Transformer;
- active weight traffic is >=3x lower;
- scaling total capacity improves quality without proportional active compute growth.

### Negative signal

The current design should be considered unsuccessful if after reasonable tuning:

- router collapses to the same circuits for all tasks;
- routing requires dense work comparable to the full model;
- quality collapses as sparsity increases;
- increasing total capacity does not improve quality unless active capacity also grows proportionally;
- recurrent state cannot reliably preserve required information;
- the model behaves like a collection of isolated experts rather than composable circuits.

Failure is useful data. Record it rather than hiding it.

---

## 12. Most important scaling experiment

After a stable V0 exists, create three models with approximately constant active compute:

```text
NE-20    20M total   ~5M active
NE-50    50M total   ~5M active
NE-100  100M total   ~5-8M active
```

Keep the benchmark comparable.

The critical question:

```text
Does quality increase substantially as total capacity grows,
while active computation remains nearly constant?
```

If yes, this is evidence that **capacity can be partially decoupled from compute**.

That result is more important than raw token/s at this stage.

---

## 13. Training strategy for RTX 3060 12 GB

Prioritize rapid iteration.

Recommended defaults:

- PyTorch;
- CUDA;
- BF16 if supported reliably, otherwise FP16;
- AdamW initially;
- gradient clipping;
- modest sequence lengths;
- dynamic synthetic batches generated on CPU;
- deterministic validation sets;
- checkpoints only for useful runs.

Target experiment classes:

### Smoke test

```text
1-5 minutes
```

Purpose:

- code correctness;
- gradient flow;
- routing not NaN;
- loss decreases.

### Short research run

```text
10-30 minutes
```

Purpose:

- reject bad ideas quickly;
- compare architectural variants.

### Serious V0 run

```text
~1-3 hours if needed
```

Purpose:

- reliable comparison to baseline;
- scaling/ablation evidence.

Do not spend days training one architecture before it has passed short runs.

Actual timing depends on implementation efficiency; record measured times rather than assuming them.

---

## 14. Experimental discipline

Every run should save a compact JSON or CSV record.

Recommended fields:

```text
run_id
commit_sha
model_name
seed
total_params
active_params_mean
active_params_max
router_type
num_circuits
active_circuits
state_dim
internal_steps
batch_size
training_examples
training_seconds
peak_vram_mb
train_loss
val_loss
exact_accuracy
samples_per_second
notes
```

Do not rely on terminal history.

Create a reproducible command for every meaningful result.

---

## 15. Suggested repository structure

The implementing agent may adjust this, but keep the project easy to inspect.

```text
Neural-Engine/
├── README.md
├── requirements.txt
├── configs/
│   ├── ne_v0.yaml
│   └── transformer_30m.yaml
├── neural_engine/
│   ├── __init__.py
│   ├── model.py
│   ├── state.py
│   ├── router.py
│   ├── circuits.py
│   └── instrumentation.py
├── baseline/
│   └── transformer.py
├── data/
│   ├── generator.py
│   └── tasks.py
├── train.py
├── evaluate.py
├── benchmark.py
├── tests/
│   ├── test_router.py
│   ├── test_circuits.py
│   ├── test_data.py
│   └── test_forward.py
└── results/
    └── README.md
```

---

## 16. Development milestones

### Milestone 0 — harness

Implement:

- deterministic synthetic task generator;
- train/validation/test split logic;
- metric logging;
- GPU timing;
- parameter counting;
- baseline Transformer.

Acceptance:

- baseline learns several task families;
- one command trains it;
- benchmark numbers are logged.

### Milestone 1 — minimal Neural Engine

Implement:

- input encoder;
- persistent state;
- circuit bank;
- simple router;
- fixed number of internal steps;
- output head.

Acceptance:

- gradients reach routed circuits;
- loss decreases;
- model solves at least simple tasks.

### Milestone 2 — true hierarchical routing

Replace any temporary full dense router with a cheap hierarchical address mechanism.

Acceptance:

- routing cost grows sublinearly with circuit count;
- active fraction can be measured;
- circuits do not trivially collapse.

### Milestone 3 — multi-skill composition

Train mixed tasks and composed tasks.

Acceptance:

- different circuit combinations emerge;
- held-out compositions are tested;
- compare to Transformer.

### Milestone 4 — capacity/compute scaling

Train NE-20 / NE-50 / NE-100 with similar active compute.

Acceptance:

- produce a table and plots for quality vs total params and quality vs active params.

### Milestone 5 — hardware optimization decision

Only now decide whether custom C++/CUDA kernels are justified.

Proceed if algorithmic memory traffic is strongly lower but PyTorch wall-clock speed fails to reflect the advantage.

---

## 17. Ablations that must eventually be run

To know what actually matters, compare:

1. hierarchical router vs dense router;
2. recurrent state vs no state;
3. few large circuits vs many small circuits;
4. fixed top-k vs variable activation budget;
5. fixed internal depth vs adaptive depth;
6. random routing baseline vs learned routing;
7. shared circuits vs task-separated circuits;
8. 10%, 20%, 40%, 100% active capacity;
9. total capacity scaling at fixed active capacity.

---

## 18. Anti-cheating rules for the architecture

A result is not meaningful if hidden dense work dominates the system.

When calculating active compute, include:

- router computation;
- state updates;
- candidate scoring;
- selected circuits;
- output computation;
- any auxiliary learned modules used at inference.

Do not claim `5M active parameters` if a hidden 25M dense controller runs first.

Likewise, do not treat inactive stored parameters as free if they must all be read to choose the active set.

The routing/search mechanism is part of the inference cost.

---

## 19. Long-term hardware hypothesis

This project is motivated partly by the observation that modern LLM inference is often heavily constrained by memory bandwidth.

If future versions can reduce the active working set by one or two orders of magnitude, a possible hardware model becomes:

```text
Large parameter store -> ordinary system RAM
Small active working set -> cache / local fast memory
Routing/control -> CPU
Neural compute -> CPU/NPU/GPU depending on platform
GPU remains available for graphics or other workloads
```

Do not optimize for this hardware model in V0. First prove the architectural principle.

---

## 20. Research references / related directions

Useful concepts to study for comparison, not copying wholesale:

- conditional computation;
- dynamic sparsity;
- contextual sparsity;
- PowerInfer-style neuron activation observations;
- DejaVu/contextual sparsity;
- product-key memories;
- adaptive computation time;
- recurrent/state-space models such as Mamba;
- mixture-of-depths;
- sparse memory systems;
- learned indexing and approximate nearest-neighbor search.

The purpose of reviewing these works is to identify solved subproblems and avoid rediscovering known failures.

The project should still maintain its central distinction:

> **many composable micro-circuits forming a dynamic computation graph, rather than selecting a few complete experts.**

---

## 21. Instructions for the implementation agent

If you are the agent running on the RTX 3060 desktop, proceed without waiting for architectural perfection.

### First implementation order

1. Create the repository structure.
2. Implement synthetic task generation.
3. Implement a competent ~20-30M Transformer baseline.
4. Confirm baseline training and benchmark instrumentation.
5. Implement a minimal Neural Engine with a temporary simple router if necessary.
6. Verify that it learns.
7. Replace the temporary router with hierarchical routing.
8. Measure actual active parameter count and touched parameter bytes.
9. Run the first controlled comparison.
10. Commit code and write results into `results/`.

### When making architecture decisions

Prefer:

- simplest falsifiable version;
- measurable behavior;
- fast iteration;
- explicit instrumentation;
- reproducible experiments.

Avoid:

- premature CUDA optimization;
- giant datasets;
- natural language before synthetic success;
- increasing parameter count to hide architectural weakness;
- undocumented changes between benchmark runs.

### Communication through GitHub

This repository is also the handoff channel between agents/machines.

For every meaningful experiment, commit:

- code/config changes;
- exact run command;
- summarized results;
- failures and anomalies;
- next recommended experiment.

Prefer Markdown reports under `results/` for important milestones.

If an architectural decision changes the project direction, document the reason explicitly before proceeding.

---

## 22. First concrete experiment

Build these two models first:

### Baseline-A

```text
Type: Transformer
Total params: approximately 20-30M
Active params: effectively dense
Dataset: synthetic mixed algorithmic tasks
```

### NE-V0

```text
Type: Neural Engine
Total params: approximately same as Baseline-A
Target active params: <=20-25% of total per internal step
State: recurrent/persistent
Routing: initially simple, then hierarchical
Circuits: many small reusable blocks
Internal steps: fixed small number for V0
Dataset: exactly same generator/distribution as baseline
```

Train both long enough to establish a stable comparison, but use short runs first.

Produce:

```text
results/V0_BASELINE_COMPARISON.md
```

with at least:

```text
- commit SHA
- GPU
- training duration
- total parameters
- average active parameters
- peak VRAM
- validation accuracy
- validation loss
- task-level accuracy
- samples/sec
- inference latency
- routing statistics
- conclusion: FAIL / WEAK SIGNAL / STRONG SIGNAL
```

---

## 23. Success philosophy

The first version does **not** need to be faster than optimized Transformer CUDA kernels.

The first version does **not** need to speak language.

The first version does **not** need 100x improvement.

The first breakthrough would be much simpler:

> A model with the same order of total learned capacity performs comparably while touching only a small fraction of its parameters, and increasing dormant capacity improves useful behavior without proportional growth in active compute.

If that happens consistently, the next research problem is routing quality and memory layout.

If that also works, custom kernels and much larger models become justified.

---

## 24. Current status

```text
Status: V0 implemented; first CUDA controlled comparison completed
Primary hardware: RTX 3060 12 GB
Immediate task: improve multi-step composition and router coverage at scale
```

Quick start:

```powershell
python -m pip install -r requirements-cuda.txt
python -m pytest -q
python train.py --model ne --steps 5000 --device cuda --balanced-train --run-id ne_v0
python train.py --model baseline --steps 5000 --device cuda --balanced-train --run-id baseline
```

The first controlled result is documented in
`results/V0_CUDA_CONTROLLED_COMPARISON.md`. It shows comparable balanced
accuracy with an estimated 4.98% active parameter fraction for NE-V0. This is
an architectural signal, not a final claim: multi-hop composition, exact total
parameter matching, and hardware-level sparse byte accounting remain open.

The subsequent NE-20/NE-50/NE-100 scaling result is documented in
`results/CAPACITY_SCALING.md`. NE-20 and NE-50 improve quality at a fixed
estimated active budget; NE-100 currently plateaus because router coverage does
not yet scale with the circuit bank.

NE-V0.2 also includes an optional two-address router. Its coverage and tradeoff
are recorded in `results/V0_2_MULTI_ADDRESS.md`.

NE-V0.3 adds an attention-free structured slot encoder and is now the main
variant. Its fixed-active NE-20/50/100 result is recorded in
`results/V0_3_SLOT_SCALING.md`.

The five-step recurrent ablation was rejected and is recorded in
`results/V0_4_FIVE_STEP_ABLATION.md`; simply increasing internal depth did not
improve composition.

The V0.5 intermediate-supervision ablation was also rejected: exposing partial
task results to recurrent steps increased training cost without improving final
or depth-3 accuracy. The controlled result is recorded in
`results/V0_5_INTERMEDIATE_SUPERVISION.md`; V0.3 remains the reference model.

Composition-focused sampling improves depth-3 accuracy only by trading away
depth-1 accuracy, so uniform task-balanced training remains the default. The
tradeoff is recorded in `results/TRAINING_SAMPLING_ABLATION.md`.

V0.6/V0.7 task-context routing was also rejected because the small overall gain
did not improve composition. See `results/V0_6_TASK_CONTEXT_ABLATION.md`.

V0.8 serial circuit composition was rejected as the default: it slightly helps
depth-2, but is about 2.1x slower and weakens depth-3. The optional mode and
controlled result are recorded in `results/V0_8_SERIAL_CIRCUITS.md`.

The held-out benchmark is now available in `results/HELDOUT_COMPOSITION.md`.
It shows that the original in-distribution score should not be treated as
systematic generalization: NE-V0.3 reaches 49.95% on held-out combinations,
roughly matching the Transformer’s 50.81%. The next focus is adaptive execution
and per-input active-compute measurement.

Adaptive halting is implemented and evaluated in
`results/V0_10_11_ADAPTIVE_HALTING.md`. V0.11 executes 1.60 of 3 steps on
average, saving about 46.7% of recurrent routed execution at a 0.57-point
balanced-accuracy cost. It is optional; V0.3 remains the default.

On held-out combinations, V0.11 reaches 47.58% while retaining the same 1.60
average steps; this is tracked in `results/HELDOUT_COMPOSITION.md`.

V0.9 structured numeric value encoding reaches 71.59% balanced accuracy and
57.11% on held-out combinations; see `results/V0_9_NUMERIC_ENCODING.md`.

V0.12 combines numeric encoding with learned adaptive halting and is now the
recommended quality/active-compute variant: 71.98% balanced accuracy with an
average of 1.60/3 recurrent steps. Its report is
`results/V0_12_NUMERIC_ADAPTIVE.md`. V0.3 remains the simpler lookup-embedding
baseline, and V0.9 remains the maximum-quality held-out reference.

Checkpoint-backed inference was then measured on the trained NE-V0.9,
NE-V0.12, and dense Transformer weights. The result is documented in
`results/CHECKPOINTED_INFERENCE.md`: V0.12 reaches 15,382 samples/s versus
3,761 samples/s for the dense control at batch size 128, while its analytical
parameter-read proxy is 16.9% of the dense model. The MAC and byte figures are
explicit estimates; the measured GPU latency is reported separately.

The trained-router utilization check is recorded in
`results/ROUTE_STABILITY.md`. V0.12 uses 96.4% of its circuit bank with no
always-hot circuit group, and reduces mean task-union route overlap from
31.22% (V0.9) to 19.43%; controlled counterfactual route pairs are the next
test before kernel-level optimization.

The controlled counterfactual test is now recorded in
`results/COUNTERFACTUAL_ROUTE_SENSITIVITY.md`: changing one operand changes
about 81% of the route set, while changing only the operation token leaves
about 2% overlap. Route replay/ablation is the next causal test.

Route replay now supplies that causal check in
`results/ROUTE_REPLAY_CAUSALITY.md`: forcing a mismatched circuit path raises
loss by 0.028–0.063 and lowers accuracy by roughly 1–2 percentage points.

Route-swap dose-response is recorded in `results/ROUTE_SWAP_ABLATION.md`:
replacing 100% of routes lowers accuracy by 1.35–1.98 points and raises loss
monotonically as the replacement fraction increases.

The active circuit budget sweep is recorded in
`results/ACTIVE_CIRCUIT_BUDGET.md`: k=4 is the fastest setting (15,457/s) with
only a 0.21-point full-benchmark cost versus k=8, while k=16 adds no quality.
The follow-up two-seed scratch validation is recorded in
`results/V0_12_MULTI_SEED_BUDGET.md`: k=8 is now the safer default because its
routers leave only 1.85–2.63% dead circuits, versus 17.6–21.2% for k=4. k=4
remains an optional low-compute mode.

V0.13 adds an optional low-temperature, coverage-aware router regularizer for
k=4. Across two seeds it reduces dead circuits to 13.1–14.8% and raises mean
validation accuracy by 0.65 points, but mean fresh full accuracy is 0.21
points lower and held-out accuracy is effectively unchanged. The result is
recorded in `results/V0_13_COVERAGE_REGULARIZER.md`; k=8 remains the default
and the new loss is disabled unless explicitly configured.

V0.14 re-tests intermediate stage supervision on V0.12. It raises mean
depth-3 accuracy from 33.33% to 36.46%, but lowers mean full accuracy from
72.19% to 71.35% and mean held-out accuracy from 72.24% to 71.46%. It remains
an optional composition-focused recipe, documented in
`results/V0_14_STAGE_SUPERVISION.md`, not the default training loss.

V0.15 scales the numeric/adaptive model from NE-20 to NE-100 with eight active
circuits. Total capacity grows 4.96x while the average active path stays near
2.04M parameters (10.07% → 2.03% active fraction); NE-100 reaches 72.92% on
the fresh held-out split on seed 17, and 73.54% full accuracy / 72.60%
held-out on the second seed. The two-seed NE-20/NE-100 mean improves by 0.26
points full and 0.52 points held-out while the active path remains constant.
The result is promising but still needs more seeds and composition tasks;
details are in `results/V0_15_NUMERIC_CAPACITY_SCALING.md`.

V0.16 tested reducing the encoded-input signal injected into later recurrent
steps. Half reinjection changes full accuracy from 71.98% to 71.67% and
held-out accuracy from 72.71% to 72.29%, so the default remains full
reinjection. The negative ablation is recorded in
`results/V0_16_INPUT_REINJECTION.md`.

V0.17 tested an explicit learned memory/write gate around the GRU proposal. It
raises held-out accuracy to 73.23% but lowers full accuracy to 70.73%, raises
the active fraction to 11.36%, and lowers throughput to 12,590 samples/s. It
is therefore not enabled by default; see `results/V0_17_GATED_MEMORY_WRITE.md`.

V0.18 adds a dedicated two-operation arithmetic composition benchmark. At
5,000 steps, both NE and the dense control are still underfit even when all
nine operation pairs are visible; NE nevertheless reaches 20.83% versus the
Transformer's 11.81% on the all-pairs control while using an estimated 10.30%
active parameter fraction. This is a useful efficiency/learnability signal,
not yet a definitive held-out generalization win. Details and the staged path
to a 300–500M feasibility test are in
`results/V0_18_COMPOSITION_BENCHMARK.md`.

V0.19 applies the same structured numeric/Fourier value frontend to the dense
Transformer baseline. On the existing held-out operand-combination protocol,
the fair Transformer reaches 72.37% train-split and 65.91% held-out accuracy,
versus NE-V0.12's 71.41% and 55.68%. The old 51.02% Transformer result used a
plain embedding and is no longer a fair quality control. NE retains the
active-compute and throughput advantage; the complete correction is recorded
in `results/V0_19_FAIR_NUMERIC_BASELINE.md`.

V0.20 adds a batched route-indexed `LazyAdamW` prototype. At 20M it stays
within 0.32 held-out points of dense AdamW and is about 4.7% slower, while
avoiding a full-bank optimizer update on each step. It is still experimental:
RAM offload, GPU circuit caching, and H2D measurements remain open. It also
passes a 300M early screen with nearly identical 200-step quality and lower
peak VRAM. See `results/V0_20_LAZY_ADAMW.md`.

V0.21 adds an inference-only CPU-RAM circuit cache. On the 100M checkpoint, a
full working-set cache reaches 97.61% hit-rate and reduces measured H2D traffic
to 167 MB over 100 batches, but runs at 9,422 samples/s versus roughly 14.6k/s
for the GPU-resident path. Pinned memory, asynchronous prefetch, and batched
row packing remain before training offload; see
`results/V0_21_CPU_CACHE_PAGING.md`.

V0.22 tests the sparse path at 300M and 500M. The 300M model reaches 56.80%
after 5,000 steps with 1.60 average executed steps and 0.68% average active
parameters. The 500M model fits on a 12 GiB RTX 3060, but reaches 44.32% with
plain routing and 44.11% with a coverage regularizer; the larger circuit bank
is therefore hardware-feasible but not yet a useful quality scaling step. See
`results/V0_22_SCALE_300M_500M.md`.

V0.23 tests parent-based capacity growth. New 500M circuit rows are cloned
from heavily used rows in the trained 300M parent, and the expanded routing
geometry is enabled after a warm-up. Across two seeds, mean held-out accuracy
reaches 52.34% versus 45.63% for scratch 500M, while mean full accuracy is
51.20%. This is a positive capacity-conversion result, but the 300M parent at
56.80% remains stronger and the growth run is not a scratch A/B. See
`results/V0_23_CAPACITY_GROWTH.md`.

V0.24 adds a third growth seed. The three-seed mean is 51.91% ± 0.83 held-out
accuracy and 51.14% ± 0.54 full accuracy, versus 45.63% and 44.32% for the
scratch 500M reference. The result confirms the direction of the growth
benefit, but remains a warm-start experiment; composition/generalization is
the next falsification. See `results/V0_24_GROWTH_THIRD_SEED.md`.

Do not assume any result beyond the recorded measurements.

This repository should preserve negative results as carefully as positive ones.
