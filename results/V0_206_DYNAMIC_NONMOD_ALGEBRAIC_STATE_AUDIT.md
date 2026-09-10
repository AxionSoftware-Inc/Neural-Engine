# V0.206 Non-Modular Algebraic State Primitive

## Question

Can a compact reusable algebraic state representation repair the depth-4
composition failure without increasing the circuit bank or router budget?

## Architecture

The DynamicRegister keeps its learned dense accumulator, router, factorized
circuit bank, and factorized digit output. In parallel it carries a two-scalar
packet containing normalized `x` and `x^2`, where `x` is the current ordinary
integer accumulator. The packet transition applies the known operation
semantics for add, subtract, and multiply; a learned `2 -> state_dim`
projection makes the packet available to the learned query and output state.
The semantic transition is fixed, so this experiment is a representation
diagnostic and not evidence that the model learned arithmetic from scratch.
The circuit path remains active and unchanged.

## Protocol

The matched factorized 300M-class configuration uses ordinary integer
arithmetic (`modulus: null`), operands `0--7`, train depths `1--2`, held-out
depths `3--4`, target offset `4096`, factorized `32,768`-class output, batch
size `512`, and `3,000` CUDA steps. Seeds 17 and 18 use the same protocol as
the V0.199 factorized control. The only model change is
`algebraic_state_mode: polynomial2` with `algebraic_state_value_scale: 4096`.

## Results

| variant | seed | held-out | depth 3 | depth 4 | train accuracy | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|
| factorized control | 17 | 71.09% | 82.81% | 59.38% | 100.00% | 7.35M | 2.05M |
| algebraic state | 17 | 79.20% | 87.11% | 71.29% | 100.00% | 7.35M | 2.05M |
| factorized control | 18 | 72.85% | 85.16% | 60.55% | 100.00% | 7.35M | 2.05M |
| algebraic state | 18 | 77.83% | 85.55% | 70.12% | 100.00% | 7.35M | 2.05M |
| **paired mean delta** | — | **+6.55 pp** | **+2.34 pp** | **+10.74 pp** | — | — | — |

The algebraic treatment mean held-out accuracy is `78.52%`, versus `71.97%`
for the paired factorized control. The added learned projection contributes
only `1,152` parameters at this state width (the exact total is included in
the JSON reports); total and active estimates remain effectively unchanged
at `7.35M` and `2.05M`. Factor-row utilization remains broad, so the gain is
not a router starvation artifact.

## Interpretation

This is the first large, repeatable signal on the current non-modular depth-4
gate. The improvement is concentrated exactly where the state-trace audit
found failure: later composition stages. The learned dense writer no longer
has to reconstruct the entire numeric identity from a drifting hidden vector;
it receives a stable algebraic coordinate packet.

The result is not yet a production adoption result. The packet uses fixed
ordinary integer operation rules and is bounded by the chosen normalization
scale, so the next gate is operand-range stress and then an ablation that
measures how much learned circuit computation remains necessary. No 500M,
700M, or 1B scaling follows until the signal survives that stress.

## Decision

**POSITIVE ARCHITECTURE SIGNAL; RETAIN AS THE LEADING P-004 HYPOTHESIS.** Do
not replace the default Native Engine yet. Keep the primitive opt-in, test it
on operands `0--15`, and report exact output cost separately. If the wider
range collapses, the current gain is only a bounded-task prior; if it holds,
it is evidence for a reusable algebraic state interface rather than a router
capacity fix.

## Artifacts

- `neural_engine/dynamic_register.py`
- `train_dynamic_composition.py`
- `tests/test_dynamic_register.py`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_algebraic_state.yaml`
- `results/runs/nonmod_depth4_values0_7_factorized_algebraic_state_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_7_factorized_algebraic_state_seed18_3000.json`
