# V0.76 Operation-Specific Circuit Banks at 9,000 Steps

## Purpose

V0.75 produced a consistent +9.03-point gain on the 3,000-step broad-value
screen. This follow-up checks whether the gain survives a longer optimization
budget or disappears as the shared-bank and operation-bank models converge.

The model and benchmark are unchanged: three operation-specific factorized
circuit banks, shared router/state/value encoder, typed-write adapter, broad
values `0..7`, and held-out `add -> multiply` / `multiply -> add` orders.

## Reproduction

Base commit: `8eb106c` (`add operation specific circuit banks`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed17_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed17_9000.pt --examples-per-task 1024 --log-every 1000
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed18_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed18_9000.pt --examples-per-task 1024 --log-every 1000
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 80.6641% | 92.0898% | 69.2383% | 7,398,279 | 1,896,352 |
| 18 | 100.0000% | 78.4180% | 91.0156% | 65.8203% | 7,398,279 | 1,896,352 |
| **mean** | **100.0000%** | **79.5410%** | 91.5527% | 67.5293% | — | — |

The V0.69 shared-bank typed-write baseline reaches 73.3398% mean at 9,000
steps. Operation-specific banks therefore retain a +6.2012-point advantage
after the longer budget. Relative to the V0.75 3,000-step operation-bank mean
of 78.1982%, the longer run adds 1.3428 points.

The stored model has 7,398,279 parameters and the active estimate is 1,896,352
(25.63%). The extra stored banks increase capacity without making all three
primitive banks active on a two-operation program.

## Decision

The operation-specific bank direction is accepted for the next development
stage. The result is no longer a one-seed or short-budget artifact. The
remaining weakness is asymmetric: `add -> multiply` is above 91.5% mean, while
`multiply -> add` remains 67.5%. The next measurement should focus on why the
second operation order is harder and on extending the same bank mechanism to
longer non-modular programs before considering a larger virtual bank.
