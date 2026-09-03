# V0.64 Zero-Initialized Gated Adapter

## Question

V0.63's operation-conditioned adapter improves prior-free non-modular
composition but hurts the mod-32 modular-template gate. This experiment adds
a scalar `tanh` gate initialized at exactly zero. The intent is to preserve
the baseline path initially and let optimization open the adapter only when
it helps.

## Results

### Prior-free non-modular composition

| Variant | Seed | Held-out mean | Gate value at end |
|---|---:|---:|---:|
| Fixed operation adapter | 17 | 84.18% | 1.00 |
| Fixed operation adapter | 18 | 84.18% | 1.00 |
| Gated operation adapter | 17 | 80.32% | +0.116 |
| Gated operation adapter | 18 | 82.23% | −0.114 |
| **Two-seed mean, fixed** | -- | **84.18%** | -- |
| **Two-seed mean, gated** | -- | **81.27%** | -- |

The gate opens only partially and chooses opposite signs across seeds. It
preserves an improvement over the 77.86% baseline, but loses 2.91 points
relative to the fixed adapter.

### Mod-32 modular strict gate

| Variant | Seed 17 | Seed 18 | Two-seed mean |
|---|---:|---:|---:|
| Templates, no adapter | 96.04% | 95.02% | 95.53% |
| Templates + fixed adapter | 94.41% | 93.60% | 94.01% |
| Templates + gated adapter | 95.34% | 94.36% | 94.85% |

The gate reduces the fixed adapter's regression by about half, but does not
fully recover the adapter-free modular reference. Its final values also have
opposite signs (`−0.143` and `+0.154`), so the learned scalar is not a
stable cross-seed policy.

## Interpretation

Zero initialization successfully prevents an immediate adapter perturbation,
but a single global scalar is too weak to decide when the adapter should be
used. It also creates a slower optimization path on the prior-free task:
the adapter weights receive no useful gradient until the gate opens.

This is a useful safety mechanism, not the current best architecture. The
evidence supports two explicit lanes for now: a fixed operation adapter for
prior-free continuous composition and no adapter for the modular-template
arithmetic path.

## Decision

1. Reject the gated adapter as the quality reference for both gates.
2. Keep it as an optional safety ablation for future multi-task routing.
3. Use the fixed operation adapter in the prior-free continuous lane.
4. Use the adapter-free modular-template configuration as the modular
   reference because it is more stable across moduli.
5. Do not increase capacity yet. The remaining issue is conditional interface
   selection, not a lack of virtual circuits.

## Run records

- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_gated_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_gated_seed18_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_operation_adapter_gated_mod32_strict_seed17_9000.json`
- `results/runs/ne_dynamic_300m_depth8_modular_templates_operation_adapter_gated_mod32_strict_seed18_9000.json`
