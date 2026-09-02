# V0.54 trainable modular template screen

Status: **positive synthetic screen; conditional promotion pending a second
modulus and sparse-residual integration**

## Architecture

`TrainableModularTemplateRegister` is an attention-free recurrent register.
Its state is a 64-way value distribution. The three primitive actions are
implemented by modular-equivariant wiring:

- addition: circular translation;
- subtraction: inverse circular translation; and
- multiplication: modular action on the state support.

There is no stored dense 3×64×64 transition parameter table. The only learned
operator-specific part is a 3×3 token-to-primitive template matrix (9
parameters); the output head is a small 64×64 linear readout. This is a
structured architecture screen, not a claim that the model discovers modular
arithmetic from an unconstrained neural circuit bank.

## Protocol

Train on depths 1–4 and values 0–31. Evaluate on unseen depths 5–6 and values
32–63. Each run uses 3,000 steps, batch size 512, and 512 evaluation examples
per depth.

## Results

| Seed | Train accuracy | OOD accuracy | Total parameters | Time |
|---:|---:|---:|---:|---:|
| 17 | 100.00% | 100.00% | 4,169 | 53.1s |
| 18 | 100.00% | 100.00% | 4,169 | 53.1s |

The template matrix converged to the identity mapping in both runs, with
diagonal probabilities around 99.54%. The two-seed OOD mean is **100.00%**.
This is a large improvement over the learned Dynamic Register reference at
about 34.6–34.8% on the same gate.

## Interpretation

The result confirms that the missing extrapolation can be supplied by a
compact, trainable modular primitive interface without attention, a dense
transition table, or a large model. It also shows why simply increasing the
virtual circuit bank was ineffective: the learned bank had no equivariance
constraint tying values together.

The result remains conditional. The primitive wiring currently encodes the
mod-64 arithmetic family, so it is a structured control rather than evidence
of general-purpose arithmetic learning. The model also does not yet contain
the large sparse factorized circuit bank. The next test must combine this
interface with trainable sparse residual circuits and report whether the
residual path adds useful behavior without breaking OOD generalization.

## Next acceptance gate

1. add the template register as the compact working-state interface to the
   Dynamic Register Machine;
2. keep shared factor mixing and the fixed active-circuit budget;
3. compare no-residual, random-residual, and template-initialized residual
   variants;
4. repeat the value/depth OOD gate on two seeds; and
5. repeat the same architecture at another modulus or with a renamed
   primitive vocabulary before promoting it as a general Neural Engine path.

## Reproduction run IDs

```text
ne_modular_templates_unseen_values_seed17_3000
ne_modular_templates_unseen_values_seed18_3000
```
