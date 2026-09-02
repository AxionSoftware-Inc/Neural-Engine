# V0.53 modular prior pilot

Status: **strong synthetic result; not yet accepted as a general learned
architecture**

## Why this pilot exists

V0.50–V0.52 showed that the Dynamic Register Machine learns the visible
mod-64 arithmetic but does not extrapolate `add/subtract` reliably to values
32–63. Capacity, shared factor mixing, state width, generic write gating,
Fourier input features, rank, exploration, and value-independent routing did
not solve that gate.

This pilot tests the missing inductive bias directly.

## Controls and hybrid

The exact control uses a fixed 3-operation mod-64 transition table and a
one-hot register. It has zero trainable parameters and no attention. It is a
ceiling/control only: it is given the benchmark algebra by construction and
must not be compared to a learned model as an equal-capability result.

The hybrid adds the same fixed transition register as a modular prior to the
Dynamic Register Machine. The original recurrent register, factorized router,
shared factor-mix circuit bank, writer, and output head remain trainable. A
small trainable projection exposes the exact modular state to the recurrent
query and readout. This is a structural-prior pilot, not teacher distillation.

## Results

Protocol: train depths 1–4 on values 0–31, evaluate depths 5–6 on values
32–63, 3,000 steps, batch 512, seed 17, 20M virtual bank.

| System | Train accuracy | Unseen-value/depth accuracy | Trainable parameters | Decision |
|---|---:|---:|---:|---|
| Learned Dynamic Register, shared mix | 99.61% | 34.57% | 1.74M | reference |
| Exact modular transition control | 100.00% | 100.00% | 0 | diagnostic ceiling |
| Dynamic Register + modular prior | 100.00% | 100.00% | 1.77M | strong prior signal |

The hybrid reaches 100% in all depths on both train and unseen ranges. It
converges extremely quickly: loss is about 0.0013 at step 500 and about
0.0002 at step 1,000. Its estimated active path is 1.48M, only slightly
above the shared-mix reference, but that estimate includes the trainable
projection and does not count fixed transition wiring as parameters.

The route audit also changes materially: the modular prior causes broad factor
row coverage rather than the narrow route reuse of the learned-only model.
This is an effect to measure, not evidence that all virtual circuits became
useful; the exact prior already carries the task solution.

## Interpretation

The result strongly localizes the problem. A non-attention register can solve
the task perfectly when its state transition has the correct modular algebra.
The failure of the learned-only model is therefore not caused by the absence
of attention, insufficient virtual capacity, or an inherently impossible
OOD task. The learned circuit/register interface lacks the right algebraic
inductive bias.

The hybrid is not yet a fair claim of learned arithmetic because the fixed
transition table encodes the answer rule. It is useful as a scaffold and as a
design target: replace the hard-coded table with trainable reusable operator
templates, preserve modular equivariance, and require the same unseen-value
gate. If a trainable template model loses the 100% result, the gap measures
what must be learned rather than what was compiled in.

## Next experiment

Build a trainable modular circuit bank with fixed equivariant wiring:

1. keep a compact phase/one-hot register interface;
2. learn operation templates or low-rank residuals instead of storing a dense
   3×64×64 table;
3. preserve hard sparse routing and shared factor mixing;
4. test initialization from the fixed prior, random initialization, and a
   no-prior control on the same OOD gate; and
5. validate on a second modulus/task before calling it a general architecture.

## Reproduction

```text
benchmark_dynamic_modular_control.py
results/runs/dynamic_exact_modular_control.json
ne_dynamic_20m_modular_prior_unseen_values_3000
```
