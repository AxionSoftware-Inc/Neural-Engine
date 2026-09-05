# Results

V0.172 shows that removing the variable-`K` scale discontinuity only improves
the failed depth schedule from `+5.8722` to `+4.5923` (`17.90%` top-1); it does
not approach the gate. Manual active-count/scale tuning is therefore closed
as a path. See `V0_172_QWEN_VARIABLE_K_SCALE_AUDIT.md`.

V0.171 rejects the `6,6,6,6,4,4,4,4` depth-dependent active schedule. Giving
the early layers 75% active groups makes the first hard-training layer
unstable and worsens the eight-layer result to `+5.8722` (`6.35%` top-1), so
capacity allocation alone is not the fix. See
`V0_171_QWEN_DEPTH_ACTIVE_SCHEDULE_AUDIT.md`.

V0.170 rejects a bounded per-channel output contract. Matching sparse-child
mean/std to the parent lowers local MSE but worsens matched four-layer held-out
CE from `+0.0095` to `+0.1549` (`78.59%` top-1). Token-level residual geometry
must be preserved; marginal statistics are insufficient. See
`V0_170_QWEN_OUTPUT_CONTRACT_AUDIT.md`.

V0.169 rejects layerwise final-logit refinement. Refining each child against
the task loss while keeping the sparse prefix and dense suffix fixed worsens
the matched four-layer result from `+0.0095` to `+0.4703` (`69.43%` top-1).
The sparse interface needs a stable hidden-state contract, not only a
different training objective. See
`V0_169_QWEN_LAYERWISE_TASK_REFINEMENT_AUDIT.md`.

V0.168 rejects generic joint cascade refinement. With the same eight-layer
protocol, 50 logit-level joint steps worsen the held-out delta from `+0.5685`
to `+0.9811` (`55.18%` top-1), so a single end-to-end objective is not fixing
the depth cascade. The next experiment must constrain the interface between
adjacent sparse layers. See `V0_168_QWEN_JOINT_CASCADE_REFINEMENT_AUDIT.md`.

V0.167 finds the strongest routing result so far. An exact best-subset oracle
passes at 2 and 4 layers (`+0.0384`, `+0.0439`), proving the current cells have
headroom. A coupled 70-class subset router with its weights frozen after
supervision is reproducible at 2 layers (`+0.0109`, `+0.0415`) and at 4 layers
with a gentler hard phase (`+0.0095`, `+0.0454`). The same method still fails
at 8 layers (`+0.5685`), so do not scale yet. See
`V0_167_QWEN_COUPLED_SUBSET_ROUTER_AUDIT.md`.

V0.166 rejects an always-on residual coreset (`+0.1378`), finds best-subset
router supervision unstable (`+0.0407` vs `+0.0567` across seeds), and shows
that removing neuron `N/K` rescaling helps but still fails (`+0.2779`). The
next diagnostic is an exact group best-subset oracle before changing the FFN
representation. See `V0_166_QWEN_RESIDUAL_SUBSET_AND_NEURON_SCALE_AUDIT.md`.

V0.165 separates router error from decomposition error. Activation-signature
clustering passes one seed at `+0.0377` but fails the second at `+0.0518`;
teacher-dot supervision is also unstable (`+0.0274` vs `+0.1148`). A clean
rank-0 oracle-dot control still fails at `+0.9487`; exact best-subset follow-up
shows dot was only a bad heuristic, not an ideal group oracle. The next test is
an always-on residual coreset with signed selected-cell coefficients.
See `V0_165_QWEN_ROUTER_ORACLE_AND_CLUSTER_AUDIT.md`.

V0.164 audits teacher-derived group decoders, activation-balanced and
overlapping partitions, finer group routing, router supervision, and
sensitivity-aware active schedules. Random overlap is not reproducible
(`+0.0079` vs `+0.0983` across seeds); teacher full-output/residual decoders
fail; and eight-layer sensitivity schedules still fail (`+0.0975`). The
five-layer sensitivity subset passes at `+0.0292` but is only a partial
sparsity endpoint. See `V0_164_QWEN_ROUTING_PARTITION_FOLLOWUPS.md`.

V0.163 audits the `taklif15.md` learned latent basis. A 4-layer, 25%-active
rank-384 basis reaches `+0.1519` held-out CE delta and `18.65x` timing despite
using `0.764x` parent storage. It is a NO-GO in its current random-basis,
soft-all-bases form. See `V0_163_QWEN_LATENT_BASIS_AUDIT.md`.

V0.162 audits depth-aware schedules and leakage-free joint cascade refinement.
Eight layers still fail with uniform 50% (`+0.0854`), uniform 75% (`+0.0777`),
both depth schedules, and joint calibration refinement (`+0.0769` best). See
`V0_162_QWEN_DEPTH_SCHEDULE_JOINT_AUDIT.md`.

V0.161 adds cross-group output mixing. It passes the held-out gate on two
seeds for both 2-layer and 4-layer 50%-active replacements (`+0.0306`/
`+0.0306` and `+0.0181`/`+0.0155`). Eight layers still fail at 50% and 75%
active, while 25% active fails at four layers. See
`V0_161_QWEN_CROSS_GROUP_MIXING_SCALE.md`.

V0.160 audits scalar hard-route scaling, nonlinear macro-cells, shared-basis
correction, and late-layer scope. The best scale point (`scale=4`) reaches
`+0.0931` held-out CE delta, while rank-384 shared-basis reaches `+0.0940`;
neither passes the `+0.05` gate. The current independent-group decomposition
remains structurally lossy.
See `V0_160_QWEN_SPARSE_APPROXIMATION_ABLATIONS.md`.

V0.159 adds an individual-neuron sparse prototype. A chunked 2-layer run
reaches `+0.2949` held-out CE delta and `2.97x` timing after 100/20 training
steps; chunking fixes the large temporary allocation but not quality or
runtime. A fused selected-neuron kernel is required before more training. See
`V0_159_QWEN_NEURON_LEVEL_SMOKE.md`.

V0.158 is a held-out audit of the transferred-neuron group path. The earlier
synthetic 50%-active pass does not generalize: the same configuration reaches
`+0.2919` CE delta on `data/qwen_eval.txt`. Interleaved, norm-balanced, finer
groups, and an oracle route all fail as well. The exact dense Qwen transfer
remains valid, but do not scale the current sparse group path yet. See
`V0_158_QWEN_HELDOUT_ROUTING_AUDIT.md`.

V0.157 compares transferred-neuron dispatches: grouped gives +0.0413 CE
 delta but 1.101x timing, while token-loop gives +0.0789 and 1.031x on the
 same seed. Keep grouped as the quality reference until a fused path preserves
 route stability. See `V0_157_QWEN_TRANSFERRED_DISPATCH_COMPARISON.md`.

V0.156 promotes 16 groups/top-4 as the first stable 25%-active eight-layer
 configuration. Two seeds pass at +0.0413 and +0.0412 CE delta, with 25% active
 expert-body compute. See `V0_156_QWEN_25_ACTIVE_STABLE.md`.

V0.155 keeps grouped dispatch numerically correct (`1.68e-8` max error) and
 caches transferred slice weights, but the 8-layer 50%-active full-model
 timing remains 1.043x of the parent. `torch.compile` cannot run here because
 Triton is unavailable. See `V0_155_QWEN_GROUPED_RUNTIME.md`.

V0.154 finds a large but not yet stable 25%-active improvement: 16 groups/top-4
 reaches +0.0499 and +0.0547 CE delta across two seeds, while 8 groups/top-2
 reaches +1.3736. A contribution-norm oracle router is also negative. See
 `V0_154_QWEN_GRANULARITY_ROUTER_ABLATIONS.md`.

V0.153 rejects the first 25%-active transferred-neuron operating point:
 8 groups/top-2 reaches +1.3736 CE delta and 69.21% teacher agreement.
 A scale-E ablation is also negative, so the current stable scale remains
 E/K and the next test is finer partitioning. See
 `V0_153_QWEN_TRANSFERRED_25_ACTIVE.md`.

V0.152 is the strongest Qwen multi-layer sparse signal so far. Partitioning
 copied Qwen SwiGLU intermediate neurons into 8 groups and selecting top-4
 keeps 50% active expert compute and passes the eight-layer quality gate on
 both seeds (`+0.0129` and `-0.0041` CE delta). Current Python grouped timing
 is still about 1.054x in the repeated seed-2026 run, so performance remains
 open. See `V0_152_QWEN_TRANSFERRED_NEURON_SPARSE.md`.

V0.151 passes exact eight-layer Qwen FFN function transfer without training:
 copying gate/up/down weights into an attention-free SwiGLU child gives
 `~1e-12` local MSE, zero alpha=0 CE delta, and 99.95% teacher agreement.
 It is a conversion control, not a sparse speedup. See
 `V0_151_QWEN_SWIGLU_WEIGHT_TRANSFER.md`.

V0.150 rejects raw random-child eight-layer scaling. The best hard-route and
 joint-refined variants remain around +0.19--+0.21 CE delta, and doubling
 child width does not help. See `V0_150_QWEN_EIGHT_LAYER_CASCADE_FAILURE.md`.

V0.149 passes the four-layer Qwen rank-8 cascade gate on two seeds. Sequential
 replacement of layers 23--26 keeps alpha=0 CE deltas at +0.0298 and +0.0376,
 with 94.29% and 94.24% teacher top-1 agreement. Fifty-iteration full-model
 timings are 0.982x and 0.971x of the dense parent, the first repeatable
 positive multi-layer timing signal. The next gate is an eight-layer adjacent
 cascade with the same 0.6B teacher. See
 `V0_149_QWEN_FOUR_LAYER_CASCADE.md`.

V0.119 conditionally passes the labels-free latent-route gate on two seeds.
V0.120 conditionally passes the labels-free non-polynomial latent-route gate on two seeds.
V0.121 conditionally passes the labels-free multidimensional latent-route gate on two seeds.
V0.122 conditionally passes the learned external-memory gate on two seeds.
V0.123 conditionally passes the task-only learned-memory gate on two seeds.
V0.124 conditionally passes the 128-entity learned-memory scale gate on two seeds, with a 2--3x quality cost.
V0.125 rejects task-only 512-entity memory scaling but recovers with an explicit address objective.
V0.126 partially recovers 512-entity task-only memory with hierarchical addressing.
V0.127 rejects larger bucket counts; 32 buckets are the current 512-row sweet spot.
V0.128 keeps temperature 1.0; temperature-only tuning does not solve the 512-row address bottleneck.
V0.129 promotes a low-weight (0.1) address objective for the 512-row hierarchical memory reference.
V0.130 passes the 2048-entity capacity gate with exact learned retrieval and one active cell.
V0.131 rejects fixed per-entity memory for held-out addresses and isolates the memorization boundary.
V0.132 passes a shared coordinate-memory control on held-out addresses without per-entity bank parameters.
 The bounded stable-cell machine reaches depth-6 MSE `8.62e-8` and `1.16e-7`,
 survives paired fact swap at `1.47e-7`/`1.80e-7`, and keeps 24.6--25.5% cell
 usage without role labels. This is the first real synthetic GO for the
 `taklif22` route objective, but the polynomial basis and direct memory lookup
 remain diagnostic limitations. See
 `V0_119_LATENT_MACHINE_UNSUPERVISED_ROUTE.md`.

V0.118 gives the strongest `taklif22` synthetic signal so far, but only
 conditionally. On a bounded state task, stable `1,x,x²` cells with role
 alignment achieve about `1e-7` MSE through depth 6, survive fact swap, and
 adapt one operation while memory stays frozen. Removing role labels causes
 route collapse and depth-6 MSE 0.4633. Thus the decomposition works, but
 labels-free latent instruction discovery is still NO-GO. See
 `V0_118_LATENT_MACHINE_BOUNDED_CONTROL.md`.

V0.117 rejects the absolute next-state transition as the canonical repair.
 It improves the exact-depth-5 point to MSE 0.0613, but depth 6 jumps to
 8.2828 and in-range quality regresses to 0.3018. The depth curve confirms
 that state stability, not route usage, remains the blocker. See
 `V0_117_LATENT_MACHINE_ABSOLUTE_TRANSITION.md`.

V0.116 adds the compact dense recurrent control. It also fits in-range
 programs and uses external facts, but fails 5--6-step OOD at MSE 63.61,
 matching the routed machine's depth failure. This means router tuning alone
 is not the next move; a stable/equivariant state transition is required.
 See `V0_116_LATENT_MACHINE_DENSE_CONTROL.md`.

V0.115 passes the conditional operation-swap modularity gate. With the fact
 table frozen, adapting only cell 0 to a changed operation reduces in-range
 MSE from 0.05456 to 0.00204 in 1,000 steps, while route usage remains
 unchanged. This validates computation/memory separation locally, but does not
 repair the long-depth OOD failure. See
 `V0_115_LATENT_MACHINE_OPERATION_SWAP.md`.

V0.114 closes the current `taklif22` state/credit interventions. Training
 across 2--6 steps lowers OOD MSE only to 51.17; explicit scalar state plus
 strong role/intermediate supervision reaches 53.11; bounded LayerNorm/clip
 variants do not help. Routes stay broad and zero-memory degrades sharply, so
 the bottleneck is repeated learned cell dynamics, not dead cells or missing
 fact use. Do not scale or transfer this V1 graph yet. See
 `V0_114_LATENT_MACHINE_STATE_INTERVENTIONS.md`.

V0.113 starts the independent `taklif22` machine benchmark. The 92K-parameter
 four-cell machine fits 2--4-step programs but fails unseen 5--6-step programs
 badly (MSE 62.83), despite balanced cell usage and measurable memory use.
 This localizes the first failure to unbounded working-state drift rather than
 routing collapse. The next gate adds bounded residual updates and state
 stabilization. See `V0_113_LATENT_MACHINE_BASELINE.md`.

V0.112 screens routed child specialization for the Qwen two-layer handoff.
The best 4-expert top-2 run reaches +0.0544 CE delta, far better than the
dense-child +0.2377 failure, but the second seed is +0.0710 and top-3 is
+0.0766. This is a positive clue, not a stable GO; no runtime gain is claimed
because the research path still evaluates all experts. Qwen routing tuning is
paused in favor of the controlled `taklif22` machine benchmark. See
`V0_112_QWEN_ROUTED_CHILD_BANK.md`.

V0.111 rejects a zero-start rank-64 dense interface residual plus joint logit
 distillation: the two-layer alpha=0 CE delta is +0.2488 versus +0.2377 for
 the uncalibrated child. The joint loss is unstable, so simple dense
 calibration is not the current repair. The next direction is sparse
 functional specialization with only a small routed subset of child circuits
 active per token. See `V0_111_QWEN_INTERFACE_CALIBRATION.md`.

V0.110 closes two simple repairs for the V0.109 full-handoff failure. A
Qwen-like attention-free SwiGLU child improves alpha=0 CE only from +0.2377 to
+0.2230 at width 384, and doubling width to 768 remains +0.2277. One-child
controls still pass, so the bottleneck is successive child-interface error,
not isolated capacity or activation choice. The next gate is compact joint
state/interface calibration. See `V0_110_QWEN_GATED_CHILD.md`.

V0.109 rejects naive independent full handoff across two adjacent late Qwen
FFNs: the single-layer child controls pass, but replacing both at alpha=0
raises held-out CE by 0.2377. A 50% parent/child mixture still reaches only
+0.0279 CE delta, showing a partial handoff signal but clear representation
drift/error accumulation. The next gate is joint end-to-end logit distillation
with both parent layers frozen. See `V0_109_QWEN_TWO_LAYER_TRANSPLANT.md`.

V0.108 passes the minimal Qwen-to-Neural-Engine parent-function transplant
gate. A 384-wide attention-free child replaces one frozen late Qwen3-0.6B FFN
with alpha=0 and stays within +0.0103, +0.0394, and +0.0183 CE across two
seeds and two late layers, with 94.53%--98.44% teacher top-1 agreement. The
child uses 8.37% of the parent FFN's scalar parameters. This is a positive
local function-transfer signal, not yet an end-to-end language result; the
next gate is multi-batch two-layer replacement. See
`V0_108_QWEN_PARENT_TRANSPLANT.md`.

V0.107 passes a synthetic self-describing semantic-address audit. A coupled
descriptor reaches 100% coarse/active routing accuracy, 100% full-scan top-k
recall, and descriptor/body cosine 0.729 while evaluating 16 local dot
products instead of 64; the free-descriptor control has cosine 0.011. This
validates the routing mechanism only, not model quality. See
`V0_107_SEMANTIC_ADDRESSING_AUDIT.md`.

V0.106 rejects the canonical equal-active-budget mesoscopic MacroCell
replacement. With 64 independent `d=384,h=480,b=128` cells and top-2 routing,
train accuracy stayed at 7.13% mean after 3,000 steps and held-out logits
became non-finite. Route utilization was broad, so the failure is hard-routing
credit assignment/state stability rather than dead cells. See
`V0_106_MESOSCOPIC_MACRO_EQUAL_ACTIVE.md`.

V0.105 transfers the canonical operator-valued layer into one unchanged
300M composition transform. It reaches only 68.36% held-out mean versus the
V0.80 reference at 79.59%, while fitting train pairs at 100%; this placement
is rejected. The isolated synthetic operator gates remain positive, but no
main-model quality gain is claimed. See `V0_105_OPERATOR_VALUED_COMPOSITION.md`.

V0.104 passes the matched operator-valued teacher task. A learned `g=16,q=8`
operator basis reaches 1.42e-13 relative MSE with 6,656 trainable scalars,
beating equal-DOF global low-rank (0.9105) and block-diagonal (0.9659)
controls; full dense reaches 4.06e-13 with 147,456 scalars. This is a
structured-target result, so the next gate is an isolated replacement inside
the existing synthetic composition model. See
`V0_104_OPERATOR_VALUED_MATCHED_TASK.md`.

V0.103 passes the isolated operator-valued parameter implementation and
representation gates. Canonical `g=16,q=8` recovers a known shared target to
4.8e-6 relative error, beats a fixed-random basis, and correctly hits a 0.959
error ceiling on random dense blocks. The direction is alive for a matched
task test, but no Neural Engine quality or runtime gain is claimed yet. See
`V0_103_OPERATOR_VALUED_PARAMETER_GATE.md`.

V0.102 tests grouped sparse execution independently of model quality. Grouping
is 1.7x faster with 32-route locality but slower for high-entropy routes; a
materialized contiguous page is about 3.3x faster in this small bank at a large
memory cost. The runtime should be locality-aware rather than always grouped.
See `V0_102_GROUPED_SPARSE_EXECUTION.md`.

V0.101 adds a matched 20M operation-bank point to the depth-5--8 capability
frontier. The two-seed held-out mean is 90.04%, below 300M at 96.39%, while
500M remains flat at 96.29%. Capacity helps from 20M to 300M but is saturated
by 300M on this task; the next experiment targets grouped sparse execution,
not another scale jump. See `V0_101_CAPABILITY_FRONTIER_20M_300M_500M.md`.

Each meaningful run is saved as a JSON file under `results/runs/`. Milestone
reports should record the exact command, commit SHA, hardware, quality, active
parameter estimate, routing statistics, and whether the result is a failure or
positive signal.

V0.97 adds a nonlinear learned basis for factor-pair interactions instead of
only adding more factor addresses. It reaches 79.10% mean held-out accuracy
across two seeds, below the V0.80 300M shared-factor reference at 79.59%.
Although `multiply -> add` improves by 0.83 points, `add -> multiply` drops by
1.81 points and the active estimate grows by about 203K. The pair-basis route
is rejected; do not scale this exact design to 500M/700M/1B. See
`V0_97_FACTOR_PAIR_BASIS.md`.

V0.98 moves the rank-8 operation write adapter before the shared writer. The
two-seed mean falls to 77.78%, so changing the adapter locus does not repair
the composition boundary. V0.99 keeps the adapter only on the terminal write;
the mean falls further to 77.25%. Both placements are rejected, and the
adapter-placement hypothesis is closed. See `V0_98_PRE_WRITER_OPERATION_ADAPTER.md`
and `V0_99_TERMINAL_OPERATION_ADAPTER.md`.

V0.100 adds a shared structured scalar value lane with learned bilinear
primitive transitions. Unsupervised and stage-supervised versions, including
a low injection scale, reach only 75.05%, 75.42%, and 76.17% mean held-out
accuracy. All are below V0.80's 79.59%; the scalar lane is rejected and the
V0.80 reference is frozen for the next benchmark/architecture phase. See
`V0_100_STRUCTURED_SCALAR_STATE.md`.

V0.92 tests stable factor warm-up during 300M -> 500M parent growth: route
only through the parent 154 factors for 1,000 steps, then open all 199. The
two-seed mean is 78.56%, below V0.90 scratch 500M (78.61%), V0.91 simple
growth (78.98%), and the V0.80 300M reference (79.59%). This rejects warm-up
as the capacity fix and points to factor-bank expressivity/ordered semantics
as the next bottleneck. See `V0_92_STABLE_FACTOR_WARMUP.md`.

V0.93 gives the first and second factor slots separate reusable factor tables
so ordered addresses are no longer assembled from one shared table. The
two-seed 300M mean falls to 77.25% versus the 79.08% shared-slot baseline,
while stored parameters rise to 13.2M and active estimate stays at 1.86M.
Ordered slots are rejected; route utilization and effective gradient coverage
must be diagnosed before another capacity expansion. See
`V0_93_ORDERED_FACTOR_SLOTS.md`.

V0.94 adds query-conditioned coefficients to each selected factor row. The
two-seed 300M mean is 78.47%, below the 79.08% shared-mix baseline, despite
an active estimate of only 1.87M. This rejects the small query-factor gate as
the default. See `V0_94_QUERY_CONDITIONED_FACTOR_MIX.md`.

V0.95 audits the trained checkpoints rather than changing the architecture.
The added 500M rows are used (10--40 of 45 rows in the held-out audit), but
usage is not correlated monotonically with quality; held-out task route
unions overlap strongly. The next screen therefore uses longer programs to
stress capacity and state credit assignment before any 700M/1B run. See
`V0_95_ROUTE_UTILIZATION_AUDIT.md` and `analyze_composition_routes.py`.

V0.96 runs that depth stress: train depths 1--4, held-out depths 5--8. The
300M two-seed mean is 96.39%; 500M is 96.29% with the same 1.90M active
estimate. The harder recurrent task still gives no capacity advantage, so
the next experiment must replace the additive shared-basis pair generator
before another scale jump. See `V0_96_DYNAMIC_DEPTH_CAPACITY_SCREEN.md`.

V0.66 tests halving the sparse circuit residual in the prior-free,
non-modular composition setup. The two-seed mean is 83.11%, below the 84.18%
full-residual operation-adapter reference but above the 79.20% no-residual
control. The circuit remains useful, but residual-scale tuning is not the
architecture fix; the full-residual configuration remains the reference. See
`V0_66_HALF_RESIDUAL_ABLATION.md`.

V0.67 adds an operation-typed low-rank adapter after the shared register
writer. Three seeds reach 89.27% mean on the same prior-free, non-modular
held-out composition benchmark, a +5.09-point gain over the full operation
adapter reference at unchanged sparse routing capacity. This is the current
learned reference, pending longer-composition and broader-value validation.
See `V0_67_TYPED_WRITE_ADAPTER.md`.

V0.68 validates the same typed-write interface on unseen recurrent depths
5--8. Two 9,000-step seeds reach 99.41% mean, matching the earlier long-depth
reference while using about 1.53M active parameters. The interface improves
optimization and remains stable, but the saturated task does not yet prove a
capacity gain. See `V0_68_TYPED_WRITE_LONG_DEPTH.md`.

V0.69 expands the non-modular operand domain from 0--3 to 0--7 and raises the
output head to 512 classes. The typed-write model reaches only 73.34% mean at
9,000 steps; a hybrid Fourier screen reaches 68.92%. Both are below the
89.27% narrow-domain reference, localizing the remaining bottleneck to numeric
state representation and composition transfer rather than raw capacity. See
`V0_69_BROAD_VALUE_INTERFACE.md`.

V0.70 adds a learned structured numeric scratch state to the typed-write
model. A 16D channel reaches 70.31% mean and a 64D channel 70.26% on the
3,000-step broad-value screen, only a small gain over the 69.17% baseline.
Widening the channel does not help, so this is not the default architecture.
See `V0_70_STRUCTURED_NUMERIC_STATE.md`.

V0.71 adds an operation-conditioned rank-16 nonlinear transition immediately
before the shared register writer. On the same broad-value screen its two-seed
mean is 66.99%, below the 69.17% typed-write baseline and the 70.31% 16D
numeric-state screen. The added transition is rejected as redundant or
interfering; it does not address the fundamental numeric interface bottleneck.
See `V0_71_OPERATION_TRANSITION.md`.

V0.72 compresses the broad-value output head from 512 to 448 classes while
still covering every target in the benchmark. The two-seed mean falls to
68.60%, below the 69.17% typed-write baseline, so unused output classes are
not the primary bottleneck. See `V0_72_COMPACT_OUTPUT_HEAD.md`.

V0.73 replaces the learned class head with a scalar Gaussian distance decoder
to impose numeric ordering on outputs. The two-seed broad-value mean collapses
to 37.48% and training remains underfit, so this output geometry is rejected.
The learned class head remains the default. See
`V0_73_SCALAR_GAUSSIAN_OUTPUT.md`.

V0.74 forces routing to depend only on operation and execution step, removing
accumulator/value context from route selection. The two-seed broad-value mean
falls to 67.43%, below the 69.17% full-route baseline, so operation-only
routing is rejected as the default. See `V0_74_OPERATION_STEP_ROUTING.md`.

V0.75 gives add, subtract, and multiply separate sparse factorized circuit
banks while keeping the router and state interface shared. The two-seed
broad-value mean rises to 78.20% at 3,000 steps, a +9.03-point gain over the
shared-bank typed-write baseline. This is the strongest current architectural
signal and is being advanced to longer-depth validation. See
`V0_75_OPERATION_CIRCUIT_BANKS.md`.

V0.76 extends V0.75 to 9,000 steps. The two-seed mean reaches 79.54%, still
6.20 points above the shared-bank 73.34% long-budget baseline. The gain is
therefore stable beyond the short screen; `multiply -> add` remains the hard
order at 67.53% mean. See `V0_76_OPERATION_CIRCUIT_BANKS_LONG.md`.

V0.77 combines the accepted operation-specific banks with the earlier 16D
numeric scratch state. The two-seed 3,000-step mean is 78.34%, only 0.15
points above the bank-only 78.20% screen, so the scratch state is rejected as
extra default capacity. See `V0_77_OPERATION_BANKS_NUMERIC_STATE.md`.

V0.78 removes both operation adapters from the operation-specific bank model.
The two-seed mean falls to 75.63%, still above the shared-bank baseline but
2.56 points below the full bank+adapter model. The banks provide the main gain,
while the adapters provide useful secondary synergy. See
`V0_78_OPERATION_BANKS_NO_ADAPTERS.md`.

V0.79 halves both operation adapter ranks from 16 to 8. The two-seed
3,000-step mean rises to 79.08% while the active estimate falls by 36,864
parameters. Rank 8 is the current efficiency candidate pending a 9,000-step
confirmation. See `V0_79_OPERATION_BANKS_RANK8.md`.

V0.80 confirms rank 8 at 9,000 steps: 79.59% mean versus 79.54% for rank 16,
with 36,864 fewer active parameters. Rank 8 is therefore the current
efficiency default; the remaining weakness is the `multiply -> add` order at
68.60% mean. See `V0_80_OPERATION_BANKS_RANK8_LONG.md`.

V0.81 gives every virtual circuit address its own factor-mix coefficients.
The two-seed 3,000-step mean falls to 78.37% from the shared-mix 79.08% and
adds 141,594 stored parameters. Per-address mixing is rejected; shared mixing
remains the default. See `V0_81_OPERATION_BANKS_PER_ADDRESS_MIX.md`.

V0.82 adds LayerNorm only on the sparse circuit input path. The two-seed
3,000-step mean falls to 78.17%, below the 79.08% rank-8 shared baseline, so
generic input normalization is rejected. See
`V0_82_OPERATION_BANKS_INPUT_NORM.md`.

V0.83 adds a modest operation-and-step hint to the otherwise full router
query. The two-seed 3,000-step mean falls to 77.25%, with no improvement on
`multiply -> add`, so hybrid routing is rejected. See
`V0_83_OPERATION_BANKS_HYBRID_ROUTING.md`.

V0.84 adds zero-initialized operation-specific residuals to the factorized
router keys. The two-seed 3,000-step mean falls to 75.46% and
`multiply -> add` falls to 67.48%, so operation-specific router keys are
rejected. Keep factor-key geometry shared and investigate the state interface
between operation banks. See `V0_84_OPERATION_ROUTER_KEYS.md`.

V0.85 adds a rank-8 operation-conditioned read residual before pair formation.
The two-seed 3,000-step mean is 78.61%, below the 79.08% bank baseline, while
`multiply -> add` moves only to 69.14%. Rank 8 is rejected as the default; a
rank-16 capacity screen is retained as the final adapter test. See
`V0_85_OPERATION_READ_ADAPTER.md`.

V0.86 doubles the operation-conditioned read residual to rank 16. The
two-seed 3,000-step mean rises to 79.81% and `multiply -> add` to 69.87%, a
consistent but modest gain. Rank 16 is the provisional state-interface
candidate pending 9k-step confirmation. See
`V0_86_OPERATION_READ_ADAPTER_RANK16.md`.

V0.87 runs the rank-16 read adapter for 9,000 steps. The two-seed mean falls
to 78.52% versus the 79.59% long-training bank baseline; the small
`multiply -> add` gain does not offset the `add -> multiply` loss. The
generic read adapter is rejected as a stable default. See
`V0_87_OPERATION_READ_ADAPTER_LONG.md`.

V0.88 adds explicit predecessor-operation context with a START embedding. The
two-seed 3,000-step mean falls to 76.05%, and `multiply -> add` remains near
baseline at 68.75%, so extra operation metadata is rejected. The next work
should change the state representation itself. See
`V0_88_PREDECESSOR_OPERATION_CONTEXT.md`.

V0.89 splits the persistent state into two independently written slots. The
two-seed 3,000-step mean falls to 76.49%, so forced slot separation is
rejected. The flat state with operation-specific circuit banks remains the
reference for the next capacity screen. See `V0_89_DUAL_SLOT_STATE.md`.

V0.90 scales the accepted flat operation-bank model from 300M to 500M virtual
circuits with scratch training. The two-seed mean falls to 78.61% from 79.08%
at 300M, while the estimated active path stays at 1.86M. Scratch capacity is
rejected as a quality fix. See `V0_90_OPERATION_BANKS_500M_SCRATCH.md`.

V0.91 warm-starts 500M from the trained 300M parent using a factor census and
cloned new factor rows. The mean recovers to 78.98%, 0.37 points above 500M
scratch, but remains 0.10 points below 300M. This is only a weak optimization
signal; 700M/1B scaling remains blocked. See
`V0_91_OPERATION_BANKS_500M_GROWTH.md`.

Useful trained checkpoints and their reproducible GPU measurements are
documented in `CHECKPOINTED_INFERENCE.md`. Checkpoint binaries live only in the
local `results/checkpoints/` directory and are ignored by Git.

Router utilization and task-overlap analysis is documented in
`ROUTE_STABILITY.md`; the reproducible entry point is `analyze_routes.py`.

Controlled one-operand and one-operation route sensitivity is documented in
`COUNTERFACTUAL_ROUTE_SENSITIVITY.md`; use `analyze_counterfactual_routes.py`
to reproduce it.

The stronger causal route-replay test is documented in
`ROUTE_REPLAY_CAUSALITY.md`; use `analyze_route_replay.py` to reproduce it.

Route replacement at 0/25/50/100% global and within-task rates is documented
in `ROUTE_SWAP_ABLATION.md`; use `analyze_route_ablation.py` to reproduce it.

The active-circuit k=4/8/16 quality and latency sweep is documented in
`ACTIVE_CIRCUIT_BUDGET.md`; use `analyze_active_budget.py` to reproduce it.

Scratch training and second-seed validation for k=4 versus k=8 is documented
in `V0_12_MULTI_SEED_BUDGET.md`.

The optional coverage-aware low-k router regularizer is documented in
`V0_13_COVERAGE_REGULARIZER.md`; it improves k=4 bank utilization but does not
yet justify changing the k=8 default.

The V0.12 stage-supervision re-test is documented in
`V0_14_STAGE_SUPERVISION.md`; it improves depth-3 accuracy but lowers overall
and held-out quality, so it remains an optional composition-focused recipe.

The current numeric NE-20/NE-50/NE-100 fixed-active scaling test is documented
in `V0_15_NUMERIC_CAPACITY_SCALING.md`; it keeps the active path near 2.04M
parameters while growing stored capacity to 100M.

The recurrent input-reinjection ablation is documented in
`V0_16_INPUT_REINJECTION.md`; reducing reinjection slightly helps depth-3 but
does not improve overall or held-out quality.

The optional explicit gated memory/write ablation is documented in
`V0_17_GATED_MEMORY_WRITE.md`; it helps one held-out measurement but increases
active cost and lowers full-benchmark quality.

The new arithmetic composition benchmark and its all-pairs learnability
control are documented in `V0_18_COMPOSITION_BENCHMARK.md`. Both models remain
underfit at 5,000 steps, so the result is a diagnostic and not a final
generalization claim.

The fair numeric Transformer re-test is documented in
`V0_19_FAIR_NUMERIC_BASELINE.md`. It shows that the old plain-embedding
Transformer control understated dense quality; NE's strongest remaining claim
is active-compute/throughput efficiency, not unconditional accuracy.

The first route-indexed batched `LazyAdamW` prototype is documented in
`V0_20_LAZY_ADAMW.md`. It stays within 0.32 held-out points of dense AdamW and
is only about 4.7% slower at 20M, but does not yet implement RAM offload or
cache/prefetch. The same optimizer remains within 0.47 points of the 100M
dense reference in a single seed and passes a 300M early feasibility screen.

The CPU-RAM LRU circuit paging prototype is documented in
`V0_21_CPU_CACHE_PAGING.md`. A full working-set cache reaches 97.61% hit-rate
and reduces 100M benchmark H2D traffic from 6.99 GB to 167 MB, but still needs
pinned-memory, asynchronous-prefetch, and batched-kernel work.

The 300M/500M sparse scaling and route-coverage study is documented in
`V0_22_SCALE_300M_500M.md`. The 300M model reaches 56.80% after 5,000 steps;
500M fits on a 12 GiB RTX 3060 but reaches only 44.32% without coverage and
44.11% with coverage, so the larger bank is feasible but not yet a useful
quality scaling step.

The parent-based capacity-growth experiment is documented in
`V0_23_CAPACITY_GROWTH.md`. Cloning new 500M circuit rows from a trained 300M
parent and warming up on the parent routing geometry raises mean held-out
accuracy to 52.34% versus 45.63% for scratch 500M, across two seeds. This is
a positive warm-start signal, not yet a scratch-training comparison.

The third-seed validation is documented in `V0_24_GROWTH_THIRD_SEED.md`.
Across three seeds, parent-based growth reaches 51.91% ± 0.83 held-out
accuracy versus 45.63% for scratch 500M, confirming the direction of the
improvement while preserving the warm-start caveat.

The V0.25 composition scaling falsification is documented in
`V0_25_COMPOSITION_GROWTH.md`. On the hidden operation-order benchmark,
300M and 500M remain underfit; 500M scratch reaches 10.16% held-out accuracy,
while parent-growth reaches 9.38%. This is a NO-GO for a 700M/1B quality jump
until circuit credit assignment or composition is improved.

The V0.26 training-only route exploration test is documented in
`V0_26_ROUTE_EXPLORATION.md`. A 5% random tree-branch exploration probability
raises 300M held-out accuracy from 7.03% to 11.72% and 500M growth from 9.38%
to 10.94% in one seed under the original small random-batch evaluation. The
later deterministic-grid audit is the primary quality reference; 700M/1B
remains blocked pending stronger composition results.

The V0.27 audited curriculum scaling study is documented in
`V0_27_CURRICULUM_SCALE_AUDIT.md`. With deterministic 4,096-example-per-pair
grids, 300M curriculum averages 13.79% ± 0.96 across two seeds, while 500M
curriculum growth averages 12.77% ± 0.24. Curriculum is a weak positive, but
500M still does not beat 300M, so 700M/1B quality scaling remains blocked.

The V0.28 typed-register architecture and 20M/50M/100M/300M scale audit is
documented in `V0_28_TYPED_REGISTER_SCALE.md`. Explicit operand/partial/final
registers and typed serial circuit execution raise hidden-pair grid accuracy
to 65.56% at 20M and 89.87% at 100M; 300M reaches 89.20% and is not better
than 100M. The result is a strong architecture positive, but not a reason to
scale to 700M/1B before route sharing and a second-seed validation.

The V0.29 routing specialization and seed audit is documented in
`V0_29_ROUTING_SPECIALIZATION_AUDIT.md`. Partitioned routing reaches 87.01%
on one 20M seed but 61.46% on the second and 68.65% at 100M. A compressed
32-dimensional route context reaches 71.75% at 20M but 61.22% at 100M. These
results reject partitioning as a reliable default and keep the larger-scale
capacity claim open.

The V0.30 active-budget audit is documented in
`V0_30_ACTIVE_BUDGET_AUDIT.md`. Doubling active circuits from 8 to 16 at 20M
does not improve all-pairs accuracy (57.67% → 57.06%) or hidden-pair accuracy
(65.56% → 63.35%), and costs about 1.5× more local training time. The next
target is credit assignment and dataflow supervision, not wider active paths.

The V0.31 full-domain question audit is documented in
`V0_31_FULL_DOMAIN_QUESTION_AUDIT.md`. It evaluates identical `64^3` operand
questions without training, adds explicit `--pair` selection to the evaluator,
and shows that 100M remains the current quality winner while 300M does not
scale monotonically. It also records that the former hidden score was
post-exposure adaptation rather than strict zero-shot generalization.

V0.32 family-local routing is documented in `V0_32_FAMILY_LOCAL_ROUTING.md`.
The hard operator/stage family split was rejected at 20M because it fell below
the reference on both all-pairs and hidden full-domain quality; it was not
scaled to larger banks.

V0.33 role-anchor routing is documented in `V0_33_ROLE_ANCHORED_ROUTING.md`.
It improves 20M hidden quality but collapses onto too few coarse cells and does
not scale to 100M. V0.34 fixed role-cell routing is documented in
`V0_34_FIXED_ROLE_CELL_ROUTING.md`; it prevents anchor collapse but still loses
100M all-pairs quality.

V0.35 shared-residual routing is documented in
`V0_35_SHARED_RESIDUAL_BANK.md`; it is rejected because the common path did not
recover the 20M reference. V0.36 multiplicative register interaction is
documented in `V0_36_MULTIPLICATIVE_REGISTER_WRITE.md`; it is the current
quality reference, reaching 94.95% at 20M but 94.13% at 100M.

V0.37 factorized virtual capacity is documented in
`V0_37_FACTORIZED_VIRTUAL_CAPACITY.md`. It reaches 96.22%, 96.25%, and 96.40%
at 20M, 100M, and 300M virtual scale respectively, with roughly 1.79M active
parameters throughout. Its hidden-stage score is still non-monotonic, so
hidden composition scaling remains an open validation task.

V0.38 factor-address routing and V0.39 factor-pair bilinear routing are
documented in `V0_38_FACTOR_ADDRESS_ROUTER.md` and
`V0_39_FACTOR_PAIR_BILINEAR_ROUTER.md`. They improve the 500M global-factor
baseline only to 95.55% and 95.43%, respectively, so neither replaces the
V0.37 global factorized default.

V0.40 depth-capped routing is documented in `V0_40_DEPTH_CAPPED_ROUTING.md`.
It reaches 94.81% at 500M, only 0.09 points above the depth-6 control, so the
500M regression is not explained by tree depth alone. `taklif6.md` records the
next second-seed, longer hidden curriculum, and parent-growth validation plan.

V0.41 records the second-seed and hidden exploration audit in
`V0_41_SEED_AND_HIDDEN_EXPLORATION.md`. The 20M all-pairs mean is 96.48% across
seeds 17 and 18; hidden adaptation improves from 96.38% to 97.16% when route
exploration is disabled for the adaptation stage. The second-seed 100M and
300M controls expose larger-bank variance, with means 95.03% and 95.89%.

V0.42 records the parent-growth conversion of a trained 300M factorized bank
into 500M in `V0_42_PARENT_GROWTH_FACTORIZED.md`. Scratch 500M reaches 94.72%,
but the grown 500M model reaches 99.66% on the same full `64^3` all-pairs grid,
with an unchanged estimated active path of 1.79M parameters. Five-thousand
step hidden adaptation reaches 99.32% on the two held-out pairs and 99.77% on
all nine pairs after adaptation. This is a strong warm-start result, not yet a
multi-seed scaling law; second-seed growth and OOD composition remain next.

V0.43 completes the second-seed, OOD, and 700M feasibility follow-up in
`V0_43_OOD_AND_700M_GROWTH.md`. Seed-18 500M parent-growth reaches 99.67%
full-grid accuracy, and 500M reaches 99.68% on a true 25%-triple combination
holdout. A 700M parent-growth screen reaches 99.82% after 3k steps and 99.66%
after a clean 10k run with the same 1.79M active estimate. The unseen value
range remains difficult: training on 0–31 and evaluating on 32–63 gives only
27.35% at 300M and 28.57% at 500M. Capacity scaling is accepted for supported
and structured composition, while representation/teacher transfer is the next
generalization target.

The proposal history is indexed in the repository root: `taklif.md` is the
original scale/systems proposal, `taklif1.md` records completed experiments
and rejected variants, `taklif2.md` records family-local routing history,
`taklif3.md` records role-anchor and fixed role-cell routing, `taklif4.md`
records the tested-and-rejected shared-residual bank, and `taklif5.md` records
the next factorized-capacity and scale-invariant-address experiment.

The V0.44 Qwen-style exact FFN circuit-graft pilot is documented in
`V0_44_QWEN_FFN_EXACT_GRAFT.md`. The gated FFN conversion is exact on a
Qwen-shaped control and was subsequently confirmed on a real Qwen3-0.6B
checkpoint; the first sparse-router pilot remains only a weak signal.

The V0.45 real Qwen3-0.6B sparse FFN and minimal hybrid audit is documented in
`V0_45_QWEN_SPARSE_AND_HYBRID.md`. Exact proposal 9 passes; 25% active
proposal 10/11 routing fails the quality gate even with a local contribution
oracle, while proposal 12 has a limited middle/late-layer positive; proposal
13 scaling and deeper proposal 14 stages remain behind the canonical stop gate.

The V0.46 teacher-distilled routing and structural compression audit is
documented in `V0_46_TEACHER_DISTILLED_COMPRESSION.md`. Global-logit
distillation plus a low-rank residual helps late-layer pilots but does not
rescue full-stack 4x routing. An adaptive oracle shows that contiguous Qwen
chunks require nearly all circuits for teacher-level fidelity, while compact
nonlinear late-layer FFNs remain a conditional, not yet accepted, direction.

The V0.47 learned-basis and layer-adaptive routing audit is documented in
`V0_47_LEARNED_BASIS_LAYER_ADAPTIVE.md`. A trainable redundant circuit basis
does not preserve full-model quality under random 25% or quick 50% execution.
Per-layer sensitivity is measurable, but a fixed schedule selected on one text
variant fails on an independent variant. The next accepted test is a global
end-to-end layer gate with larger independent data, multi-seed validation, and
an actual grouped-kernel latency measurement; do not scale Qwen transfer to
1B/1.7B before that gate passes.

The V0.48 global token-adaptive layer-gate audit is documented in
`V0_48_GLOBAL_LAYER_GATE.md`. Late-layer gating preserves teacher fidelity in
the strongest short control, but only with about 1.2% overall FFN reduction;
stronger compute pressure breaks fidelity, and gating all layers fails the
teacher top-1 gate. Keep this as a conditional research direction until a
larger multi-seed corpus and structured conditional kernel demonstrate real
savings.

The V0.49 dynamic Neural Register Machine returns the main research line to an
independent attention-free architecture. A recurrent accumulator scans
variable-length operation programs and routes eight factorized circuits per
executed step. Training only on depths 1–4 and evaluating unseen depths 5–6
reaches 72.85% at the 20M virtual tier, 73.83% at 100M, and 80.86% at 300M in
seed 17; second-seed checks reach 75.49%, 72.17%, and 76.37% respectively.
The 300M tier also leads the 20M control on unseen depths 5–8 (75.68% vs
72.12%). However, a disjoint value-range test remains low at 34.77% for 20M
and 34.38% for 300M, so the next bottleneck is value-independent circuit
generalization rather than raw virtual capacity. See
`V0_49_DYNAMIC_REGISTER_MACHINE.md`; route instrumentation also shows strong
reuse of observed virtual routes rather than single-route collapse.

V0.50 isolates the next bottleneck: a 300M model trained on values 0–31
generalizes to unseen depths 5–6 at 98.44% within that range, but falls to
33.84% on values 32–63. The operation breakdown shows multiplication transfers
well while addition/subtraction do not. This is now the acceptance gate for
the proposal ablations. See `V0_50_DYNAMIC_GENERALIZATION_DIAGNOSTIC.md`.

V0.51 screens proposals 9–15 on the same OOD gate. None produces a reliable
large jump; state width, gated write, rank, and exploration are rejected.
Parallel mix is faster but not more accurate, while input reinjection gives
only a small unconfirmed gain. See `V0_51_DYNAMIC_PROPOSALS_09_15.md`.

V0.52 screens the scalable shared factor mix and then tests 300M, 500M, and
700M capacity. Shared factor mix raises the 300M depth-holdout mean from
78.62% to 83.16% across two seeds, but 500M is 83.01% and 700M regresses to
80.76%; raw capacity is not the bottleneck. The strict unseen-value gate
remains about 35%. A value-independent operation/step router also falls to
31.25%, so the remaining problem is register/circuit modular composition, not
router drift alone. See `V0_52_DYNAMIC_CAPACITY_AND_ROUTING.md`.

V0.53 tests that hypothesis with an exact fixed modular transition control and
a hybrid structural prior. Both reach 100% on the unseen-value/depth gate,
which is a strong localization signal but not a learned-generalization claim:
the control is given the mod-64 algebra. The next accepted experiment is a
trainable equivariant template bank without a dense transition table. See
`V0_53_MODULAR_PRIOR_PILOT.md` and the proposal `taklif7.md`.

V0.141 tests a naive packed/vectorized dispatch implementation for the Qwen
single-layer sparse circuit bank. It preserves numerical correctness and the
V0.138 quality gate, but isolated FFN timing becomes 9.90x slower than the
parent and end-to-end timing is about 1.12x slower. The packed path is rejected
for performance; a grouped/fused kernel with weight reuse is required before
scaling the experiment. See `V0_141_QWEN_PACKED_DISPATCH.md`.

V0.142 tests grouped selected-token dispatch. It improves isolated FFN timing
to 1.166x and 1.204x over parent across two seeds, versus 1.337x and 1.375x
for token-loop dispatch, while preserving the quality gate. It is the best
current reference path but is still not a speedup; redundant child LayerNorm
cost is the next focused test. See `V0_142_QWEN_GROUPED_DISPATCH.md`.

V0.143 removes that redundant child LayerNorm for the Qwen transplant. Across
two 100-iteration repeats, isolated FFN timing is 1.057x and 1.005x over the
parent, with end-to-end timing about 1.015x and the quality gate still passing.
This is accepted as the preferred Qwen path but not yet a measured speedup.
See `V0_143_QWEN_GROUPED_NO_NORM.md`.

V0.144 repeats the normless grouped bank at 1,024 tokens per batch. Isolated
FFN timing reaches 0.462x and 0.444x of parent across two seeds, while the
quality gate passes with CE deltas +0.0132 and +0.0140. End-to-end remains near
neutral because only one of 28 Qwen layers is replaced. This is a performance
GO for the next multi-layer gate, not yet a larger-model claim. See
`V0_144_QWEN_LARGE_TOKEN_BLOCK.md`.

V0.145 applies 50-step full-model joint logit refinement after local training
of two normless grouped top-2 children. The combined alpha=0 CE deltas are
+0.0433 and +0.0273 across two seeds, passing the composition gate where the
unrefined seed-2026 path reached +0.0993. Joint refinement is accepted as the
current two-layer composition fix; its teacher-logit memory path still needs
optimization before long runs. See `V0_145_QWEN_TWO_LAYER_JOINT_REFINE.md`.

V0.146 retests joint refinement at 8×128. Soft and hard route variants, with
learning rates 1e-3 and 1e-4, all fail seed2026 (`+0.1017` to `+0.1247`) even
after teacher-logit CPU offload. Large-block joint refinement is rejected as a
stable composition fix; the offload remains as a memory improvement. See
`V0_146_QWEN_LARGE_JOINT_STABILITY.md`.

V0.147 adds a zero-start rank-8 low-rank interface correction to each two-layer
bank. On two 8×128 seeds the combined alpha=0 CE deltas are +0.0184 and +0.0289,
with about 34.96% of parent FFN parameters and the quality gate passing. Rank 8
is accepted as the current quality default pending timing. See
`V0_147_QWEN_TWO_LAYER_CALIBRATION_RANK8.md`.

V0.148 times that rank-8 two-layer bank end-to-end. Seed 2026 is 0.963x of
parent and seed 2027 is 1.044x, with quality passing in both cases. The timing
result is therefore neutral rather than a claimed speedup; repeatable timing or
a fused grouped kernel is required before scaling to a larger teacher. See
`V0_148_QWEN_TWO_LAYER_TIMING.md`.

V0.133 replaces the fixed learned address bank with an external random
content-key lookup. With 2048 fact rows, only the first 1536 used in training,
both seeds retrieve the 512 held-out rows at 100% and reach 4.21e-6/4.27e-6
held-out MSE, while the four-cell route remains balanced and one 162-parameter
cell is active per step. This is a strong addressing control, but not yet
semantic key learning because query and fact keys are externally supplied and
exactly matched. See `V0_133_LATENT_MACHINE_CONTENT_MEMORY.md`.

V0.134 adds one shared trainable key projection before content lookup. It adds
2,048 shared weights but no row-specific address parameters; both seeds still
retrieve all 512 unseen rows at 100% with held-out MSE 4.51e-6/4.30e-6. This
supports shared key transforms while preserving the one-active-cell path. It
is still an externally paired-key control, not semantic end-to-end memory
learning. See `V0_134_LATENT_MACHINE_SHARED_KEY_PROJECTION.md`.

V0.135 aligns two different external key views with two global projections. A
fixed orthogonal transform separates query and fact representations, yet both
seeds retrieve all 512 unseen rows at 100% with held-out MSE 3.84e-6/6.05e-6.
This is the strongest addressing signal so far: shared alignment transfers
without row-specific parameters, while the active computation cell stays at
162 parameters. The key views remain synthetic, so the next gate is frozen
Qwen-derived content features. See `V0_135_LATENT_MACHINE_DUAL_VIEW_KEYS.md`.

V0.136 rechecks the single-layer Qwen parent transplant at sequence length 128
and 600 steps. Both seeds pass alpha=0 with CE deltas +0.0068/+0.0171 and
98.05%/98.44% teacher top-1 agreement. The 384-wide child remains only 8.37%
of the 9.44M-parameter parent FFN. This confirms a stable local function
transfer, but not multi-layer replacement or full Transformer removal. See
`V0_136_QWEN_PARENT_TRANSPLANT_LONG_SEQUENCE.md`.

V0.137 makes the routed Qwen evaluation path genuinely sparse: 4 experts with
top-2 dispatch execute only the selected token-expert pairs, a 50% expert-body
fraction. Accuracy remains the same as the prior audit but fails the two-layer
quality gate at CE deltas +0.0537/+0.0708. Keep the dispatch implementation,
reject this handoff quality, and make no wall-clock claim until grouped-kernel
timing is measured. See `V0_137_QWEN_HARD_SPARSE_DISPATCH.md`.

V0.138 turns the stable single-layer Qwen transplant into a 4-expert top-1
bank. Both seeds pass alpha=0 with CE deltas +0.0237/+0.0257 and
95.51%/96.39% teacher top-1 agreement, while selected-token dispatch executes
only 25% of expert bodies. This is a quality and active-path GO for one layer;
the next gate is grouped CUDA latency and independent-text validation. See
`V0_138_QWEN_SINGLE_LAYER_SPARSE_BANK.md`.

V0.139 measures that bank end-to-end with isolated synchronized CUDA timing.
Across two serial seeds, parent latency is 66.70/66.86ms and sparse-bank
latency is 68.00/68.25ms (1.020x/1.021x), so Python-level dispatch erases the
theoretical 25% expert-body reduction. Keep the quality bank, reject the
current performance implementation, and move to grouped/batched execution
before adding layers or model capacity. See
`V0_139_QWEN_SINGLE_LAYER_BANK_TIMING.md`.

V0.140 isolates the layer timing on identical captured hidden states. Parent
FFN is 0.850/0.847ms while the selected-token bank is 1.136/1.165ms
(1.337x/1.375x) across two seeds. This confirms the overhead is inside the
small-dispatch implementation, not the surrounding model. The next required
step is packed grouped matmul or a fused CUDA/Triton kernel. See
`V0_140_QWEN_BANK_ISOLATED_FFN_TIMING.md`.

V0.60 fixes the main strict-value bottleneck with a trainable modular
value-state/template interface. On values 0--31 train and unseen values
32--63 eval, the 300M Macro-enabled model reaches 96.36% mean across two
seeds at unseen depths 5--8. Disabling Macro-Cells preserves the result at
96.53% while reducing total parameters from about 10.07M to 3.29M. Learned
value embeddings and fixed Fourier features remain at 36.90% and 24.98%, so
the gain comes from the modular interface rather than generic capacity or
encoding. This is the current reference, pending a second-modulus and
non-modular-task validation. See `V0_60_MODULAR_VALUE_STATE.md`.

V0.61 tests the core register/circuit path on held-out operation
compositions with `modular_prior: false` and no Macro-Cells. Two seeds reach
98.71% mean accuracy on the unseen `add--multiply` and `multiply--add`
orders, with 99.94% mean on the seven training pairs and about 1.45M active
parameters. This validates prior-free compositional transfer inside the
mod-64 arithmetic environment, but it is not yet a genuinely non-modular
task. See `V0_61_COMPOSITION_HOLDOUT.md`.

V0.62 removes modular reduction entirely from the composition generator.
With the same prior-free DynamicRegister and two unseen operation orders, the
two-seed held-out mean falls to 77.86% despite 100% training accuracy. This
rejects the current continuous-state interface as a validated general
non-modular composition solution and confirms that V0.61's 98.71% relies on
strong mod-64 algebraic regularity. See `V0_62_NONMODULAR_COMPOSITION.md`.

V0.63 adds a small operation-conditioned low-rank adapter shared across
recurrent steps. On the genuine non-modular composition gate it raises the
two-seed held-out mean from 77.86% to 84.18% with about 38k extra
parameters. It is not a universal default: the same adapter gives 96.90% on
the mod-64 strict gate but regresses the mod-32 mean from 95.53% to 94.01%.
It is accepted for the prior-free continuous lane and kept optional for the
modular-template lane. See `V0_63_OPERATION_ADAPTER.md`.

V0.64 tests a zero-initialized learnable gate around that adapter. The gate
reduces the mod-32 regression from 94.01% to 94.85% mean, but still remains
below the 95.53% adapter-free modular reference; on prior-free composition it
reaches 81.27%, below the fixed adapter's 84.18%. The gate is therefore
rejected as the quality default and retained only as a safety ablation. See
`V0_64_GATED_OPERATION_ADAPTER.md`.

V0.65 disables only the sparse circuit residual while keeping the operation
adapter and router. The no-residual two-seed mean is 79.20%, versus 84.18%
with the full residual, so the circuit bank contributes useful computation
and should not be removed. The current bottleneck is circuit/interface
factorization, not raw virtual capacity. See
`V0_65_NO_RESIDUAL_CIRCUIT_CONTROL.md`.

V0.61 tests the core register/circuit path on held-out operation
compositions with `modular_prior: false` and no Macro-Cells. Two seeds reach
98.71% mean accuracy on the unseen `add--multiply` and `multiply--add`
orders, with 99.94% mean on the seven training pairs and about 1.45M active
parameters. This validates prior-free compositional transfer inside the
mod-64 arithmetic environment, but it is not yet a genuinely non-modular
task. See `V0_61_COMPOSITION_HOLDOUT.md`.

V0.62 removes modular reduction entirely from the composition generator.
With the same prior-free DynamicRegister and two unseen operation orders, the
two-seed held-out mean falls to 77.86% despite 100% training accuracy. This
rejects the current continuous-state interface as a validated general
non-modular composition solution and confirms that V0.61's 98.71% relies on
strong mod-64 algebraic regularity. See `V0_62_NONMODULAR_COMPOSITION.md`.

V0.56 adds a reusable Macro-Cell bank with sparse top-1 hierarchical routing.
The 256-cell screen reaches 78.56% and 75.29% across two seeds versus the
earlier 20M reference mean of 74.17%, while the active macro estimate stays
near 34K. The signal is positive but non-monotonic, so staged parent-grown
macro expansion is required before any 300M+ or billion-parameter claim. See
`V0_56_MACRO_CELL_SCALING.md`.

V0.57 validates staged parent-grown Macro-Cell expansion. A trained 16-cell
parent is expanded through 64 cells and then 256 cells, with new router levels
opened gradually. Across seeds 17 and 18, the final 256-cell models reach
99.51% and 99.71% on unseen depths 5--6, with 255/256 and 254/256 macro cells
reachable. A fair 9,000-step scratch control reaches 99.56% and 99.80%, so
staged growth is a reliable curriculum but not a final-quality capacity gain
(means 99.61% staged vs 99.68% scratch). See
`V0_57_MACRO_PARENT_GROWN.md`.

V0.58 extends the Macro-Cell gate from unseen depths 5--6 to unseen depths
5--8. At 3,000 steps the 256-cell model reaches 78.00% and 68.85% across two
seeds (73.43% mean), while 252--256 of 256 macro cells remain reachable. The
depth slope and low train accuracy at depths 3--4 indicate incomplete
convergence, not a routing-collapse explanation. Continuing the same runs to
9,000 steps raises both seeds to 99.44% mean accuracy on depths 5--8. See
`V0_58_LONG_DEPTH_MACRO_GATE.md`.

V0.59 screens a 300M-class virtual bank with the same 256-cell Macro-Cell
path. After 9,000 steps, two seeds reach 99.39% and 99.63% on unseen depths
5--8 (99.51% mean), versus 99.44% for the 20M-class control. The +0.07-point
change is negligible and active computation stays near 1.46M parameters, so
larger 500M/700M/1B banks are not justified by this gate. See
`V0_59_300M_MACRO_DEPTH8.md`.

V0.54 runs that trainable template screen on two seeds. It reaches 100% on the
same strict OOD gate with 4,169 parameters and no dense transition table.
This is the strongest current synthetic signal, but it remains conditional on
the mod-64 structural wiring until a second modulus and sparse-residual
integration pass. See `V0_54_TRAINABLE_MODULAR_TEMPLATES.md`.

V0.55 closes that validation. Random-init templates reach 100% on two seeds at
both mod-64 and mod-32 after 10k steps. At equal 10k budget, the Dynamic
Register reaches 99.71% without circuit residual and 99.51% with residual, so
the compact modular interface solves the arithmetic gate while the sparse bank
is optional. See `V0_55_TEMPLATE_RANDOM_INIT_AND_RESIDUAL_AB.md` and the next
proposal `taklif8.md`.

V0.53 tests that hypothesis with an exact fixed modular transition control and
a hybrid structural prior. Both reach 100% on the unseen-value/depth gate,
which is a strong localization signal but not a learned-generalization claim:
the control is given the mod-64 algebra. The next accepted experiment is a
trainable equivariant template bank without a dense transition table. See
`V0_53_MODULAR_PRIOR_PILOT.md` and the proposal `taklif7.md`.
