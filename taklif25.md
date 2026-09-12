# taklif25.md — Virtual Logical Neurons: more functional capacity than physical parameters

Status: **independent capacity-representation proposal; no temporary task growth or continual learning in V0**

> **SCOPE:** Test whether a compact physical parameter substrate can generate or compose a much larger number of distinct logical neurons/circuits. This proposal is about representational/function capacity, not extra reasoning time and not storing new documents.

## Hypothesis

A model with `P_physical` learned parameters may expose a much larger addressable functional bank:

`P_logical >> P_physical`

by composing reusable learned factors/operators. Example:

`C_i = Compose(B_a, B_b, B_c; alpha_i)`.

The logical units are not claimed to be independent scalar degrees of freedom. The useful claim is narrower: increasing the number/diversity of addressable logical functions can improve held-out quality while physical storage and active compute grow slowly.

## V0 experiment

Fix one physical substrate and create logical banks at increasing sizes, for example `1x, 4x, 16x, 64x` logical capacity. Keep the same training data, physical parameter count, active route width, and evaluation protocol.

Measure whether the extra logical identities become functionally distinct and causally useful rather than aliases of the same basis.

## Required diagnostics

- pairwise functional signatures on probe states;
- effective rank / diversity of logical outputs;
- dead logical units;
- route usage entropy;
- counterfactual loss when a selected logical unit is removed/replaced;
- quality as logical bank size rises;
- physical parameters and active parameters separately.

## Success gate

A positive result requires a repeatable quality increase as logical capacity increases at nearly fixed physical storage and active budget. Merely increasing the count of addresses is not success.

## Failure / stop rule

Reject if logical capacity grows but functional diversity, causal utility, or held-out quality does not. Do not rescue by increasing the physical model in the same experiment.

## Why it matters

This directly tests the idea of, for example, `100M physical parameters` supporting `300M/1B logical computational units` without pretending those logical units are fully independent parameters.