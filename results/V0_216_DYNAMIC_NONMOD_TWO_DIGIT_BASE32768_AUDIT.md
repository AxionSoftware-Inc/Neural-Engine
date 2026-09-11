# V0.216 — full-range two-digit base-32768 codec

**Date:** 2026-09-11  
**Status:** `REJECTED`

## Hypothesis

The three-digit codec might lose composition quality because it factorizes the
class into too many independent heads. A two-digit quotient/remainder codec
was tested for the full `2^30` class range, using base `32768` for both heads
and shared projection rank `128`.

## Protocol

The model body, router, data split, target offset, optimizer, 3000 steps,
batch size, and seeds were identical to the matched three-digit screen:
values `0..63`, train depths `1..2`, held-out depths `3..4`, seeds `17/18`.
Fourier base was aligned to `32768`. The only intended change was the output
codec. Total parameters were `15,724,935`; active estimate was `10,425,776`.

## Results

| Seed | Train accuracy | Held-out accuracy | Depth 3 | Depth 4 | Held-out CE |
|---|---:|---:|---:|---:|---:|
| 17 | 53.32% | 12.30% | 15.63% | 8.98% | 10.9448 |
| 18 | 50.00% | 11.52% | 14.45% | 8.59% | 9.7042 |
| **Mean** | **51.66%** | **11.72%** | **15.04%** | **8.79%** | **10.3245** |

The large 32,768-class heads did not learn the train distribution within the
budget. This is a decisive negative result, not evidence that two digits are
mathematically impossible: the head optimization and gradient scale are the
failure mode here.

## Decision

The variant is rejected. It is not a useful continuation of the full-range
experiment and will not become default. The experiment isolates a practical
constraint: keeping the per-digit class count small is more important than
minimizing the number of digit heads.

Raw runs:

- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_two_digit_base32768_fourier32768_rank128_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_two_digit_base32768_fourier32768_rank128_seed18_3000.json`
