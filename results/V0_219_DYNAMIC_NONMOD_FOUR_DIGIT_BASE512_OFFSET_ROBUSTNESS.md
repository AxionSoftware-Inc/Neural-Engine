# V0.219 — four-digit base-512 target-offset robustness

**Date:** 2026-09-11  
**Status:** `PASSED ROBUSTNESS GATE / OPT-IN CANDIDATE`

## Question

The four-digit base-512 codec is the leading full-range candidate after the
four-seed 5000-step screen. This test changes only the target offset to check
whether the gain is tied to a particular absolute class-coordinate layout.

## Protocol

- same model body, router, circuit bank, optimizer, and codec as the leading
  four-digit config;
- values `0..63`, train depths `1..2`, held-out depths `3..4`;
- output digits: four base-512 heads, shared rank `128`;
- Fourier base `512`;
- target offset changed from `1,048,576` to `2,097,152`;
- 3000 steps, batch `128`, seeds `17` and `18`;
- matched control: the original-offset 3000-step seed17/18 runs.

## Results

| Offset | Seed | Held-out accuracy | Depth 3 | Depth 4 | Held-out CE |
|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 17 | 77.54% | 81.25% | 73.83% | — |
| 2,097,152 | 17 | 77.34% | 79.69% | 75.00% | 2.4256 |
| 1,048,576 | 18 | 81.64% | 88.28% | 75.00% | — |
| 2,097,152 | 18 | 81.25% | 89.06% | 73.44% | 2.1574 |
| **Original mean** | — | **79.59%** | **84.77%** | **74.41%** | — |
| **Shifted mean** | — | **79.30%** | **84.38%** | **74.22%** | **2.2915** |

The offset shift changes mean held-out accuracy by only `−0.29 pp` and depth-4
by `−0.20 pp`. Both seeds remain near the original quality range. This is
well inside the observed seed variance and far smaller than the improvement
over the aligned three-digit control.

## Decision

The four-digit representation **passes offset robustness**. The result is not
consistent with simple memorization of the original target offset. The config
remains opt-in while the next verification checks fresh value distributions or
operation mixes; no default model is silently changed.

## Raw runs

- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_offset2097152_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_offset2097152_seed18_3000.json`
