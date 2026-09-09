# P-003 native factorized virtual-bank audit — 2026-09-10

## Question

Can the native `NeuralEngineV0` expose the same 7,552 routed circuit addresses
through reusable factor rows, so that stored capacity grows without giving
each address an isolated parameter island?  This is an opt-in architecture
test.  The existing independent bank and all default settings remain intact.

The factorized bank represents address `a + factor_count * b` by composing two
factor rows.  The factorized router scores reusable factor keys, forms a small
Cartesian product, and still executes only the configured `active_circuits`
per recurrent step.  The pair-interaction arm additionally adds a rank-4
factor-pair basis; it was tested separately because a purely additive factor
bank may not express useful combination-specific corrections.

## Protocol

- Native V0, `slot_count=5`, numeric encoding, adaptive halting, three internal
  steps, `active_circuits=8`, 7,552 virtual addresses.
- Balanced task batches, AdamW, learning rate `3e-4`, weight decay `0.01`,
  1,000 steps, the same validation protocol, and seeds 17 and 18.
- Independent control: the existing `ne_100_v12_coverage.yaml`.
- Factorized arm: `factor_count=87`, `factor_candidate_pool=8`, no coverage
  loss because `FactorizedRouter` does not implement coverage regularization.
- Pair arm: the same factorized setup plus `factor_pair_rank=4`.
- This is a screening budget, not a convergence claim.  A quality adoption
  decision requires a longer run only if the short gate is positive.

## Results

| Arm | Seed | Val CE | Exact accuracy | Time | Total params | Used circuits | Dead circuits | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Independent | 17 | 1.42919 | 58.411% | 71.89s | 100,466,025 | 6,677 | 11.59% | 1,946 MB |
| Independent | 18 | 1.40624 | 59.167% | 72.78s | 100,466,025 | 6,674 | 11.63% | 1,946 MB |
| Factorized | 17 | 1.45306 | 58.438% | 36.73s | 3,002,690 | 864 | 88.56% | 680 MB |
| Factorized | 18 | 1.49635 | 58.750% | 36.70s | 3,002,690 | 887 | 88.25% | 680 MB |
| Factorized + rank-4 pair | 17 | 1.49475 | 57.943% | 43.37s | 3,053,726 | 834 | 88.96% | 680 MB |
| Factorized + rank-4 pair | 18 | 1.50280 | 57.865% | 43.58s | 3,053,726 | 716 | 90.52% | 680 MB |

Two-seed means:

- Independent: `58.789%` accuracy, `1.41771` CE, `72.33s`.
- Factorized: `58.594%` accuracy, `1.47470` CE, `36.71s`.
- Factorized accuracy is `−0.195 pp` and CE is `+0.05699` worse than the
  independent control.  It is about `49.3%` faster, uses `97.0%` fewer stored
  parameters, and uses `65.1%` less peak VRAM.
- The factorized router touched only `~876` of 7,552 virtual IDs on average;
  the rest are not evidence of learned capacity.  Its high dead-address rate
  is therefore a routing/coverage diagnostic, not a quality win.  The
  evaluation code now reports factor-row usage separately, because virtual
  address sparsity alone cannot prove that reusable factor rows are dead.
- Adding the rank-4 pair basis made the short-screen result worse than the
  plain factorized arm by about `0.69 pp` accuracy and `0.024` CE.  It does not
  justify a longer pair-arm run under this initialization and loss.
- A corrected exploration implementation was also screened at `10%`.  It
  reached `58.294%` mean accuracy and `1.45429` mean CE, versus `58.594%` and
  `1.47470` for plain factorized; the hard-accuracy loss and unchanged high
  virtual dead rate do not pass the gate.  An earlier exploration run had a
  weight-shape bug and is explicitly not used as evidence.
- Widening the factor candidate pool from `8` to `32` reached `58.659%` mean
  accuracy and `1.46450` mean CE.  This is still `−0.130 pp` behind the
  independent control; it took `87.68s` and `2493 MB`, so it is slower and
  more memory-hungry than both the plain factorized and independent arms while
  leaving dead virtual addresses at `87.94%`.
- The factor-row diagnostic separates this virtual-address collapse from the
  reusable basis itself: in a 2-step smoke the plain arm used `86/87` factor
  rows (`1.15%` dead).  In the rank-32 pair arm, the two-seed screen used
  `64–65/87` rows (`25–26%` dead) and reached only `58.203%` mean accuracy with
  `1.51172` mean CE.  The pair basis therefore did not recover useful
  combination capacity in this protocol.

### Longer convergence check

Because the 1,000-step screen could in principle have been too early, both
plain arms were trained from scratch for 3,000 steps under the same protocol:

| Arm | Seed 17 accuracy / CE | Seed 18 accuracy / CE | Mean accuracy | Mean CE |
|---|---:|---:|---:|---:|
| Independent, 7,552 circuits | 68.542% / 0.98952 | 68.698% / 0.99082 | 68.620% | 0.99017 |
| Factorized, 7,552 virtual addresses | 66.719% / 1.04476 | 67.057% / 1.06380 | 66.888% | 1.05428 |

The gap widens to `−1.732 pp` accuracy and `+0.06411` CE for factorized.  It
does not catch up with more optimization.  The factorized arm remains about
`49.5%` faster and uses the same `680 MB` peak VRAM, but that runtime saving is
coming from its much smaller learned basis, not from a successful 7,552-address
capacity expansion.

For an additional parameter-budget control, a 1,000-step independent bank with
87 real circuits has `3,002,985` parameters and mean `58.229%` accuracy / `1.43779`
CE, while the factorized 87-row bank has `3,002,690` parameters and mean
`58.594%` / `1.47470`.  The virtual combinations provide a small early accuracy
benefit over the same-row-count independent control, but the 174-circuit
independent control reaches `58.997%` / `1.43180`, showing that the benefit is
not equivalent to the full independent 7,552-bank capacity.

## Decision

**REJECTED as a quality/capacity fix.  RETAINED as an opt-in compression and
runtime arm.**

The implementation is useful for reducing memory and training time, but this
experiment does not show that virtual address count becomes usable learned
capacity.  The same quality plateau remains: increasing the address space does
not automatically improve the difficult tasks, and the factorized router
collapses onto a small subset of virtual addresses.  The pair interaction
extension was also rejected in this first screen.

The native defaults and independent bank were not changed.  No 300M/500M
factorized expansion is justified by this result.  P-003 and P-007 remain
open; the next useful work should improve the combination representation or
final-loss alignment, not just increase the virtual address count.

## Reproduction

```powershell
python -u train.py --config configs/ne_100_v12_coverage.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_independent_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_coverage.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_independent_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_100_v12_factorized.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_factorized.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_100_v12_factorized_pair.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_pair_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_factorized_pair.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_pair_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_100_v12_factorized_explore10.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_explore10_corrected_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_factorized_explore10.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_explore10_corrected_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_100_v12_factorized_pair32.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_pair32_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_factorized_pair32.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_pair32_s18_1000 --output results/runs --seed 18
python -u train.py --config configs/ne_100_v12_coverage.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_independent_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_coverage.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_independent_s18_3000 --output results/runs --seed 18
python -u train.py --config configs/ne_100_v12_factorized.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_factorized.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_s18_3000 --output results/runs --seed 18
```

The JSON outputs are generated under `results/runs/` and are git-ignored.
