# taklif15.md — Learned latent reusable circuit basis for pretrained dense FFNs

Status: **next representation experiment; macro-cell proposal is intentionally separate**

## Muammo

Qwen3-0.6B FFNlarini contiguous neuron chunklarga exact ajratish ishladi, ammo 4x sparse selection ishlamadi. Hatto offline oracle teacher-level fidelity uchun chunklarning katta qismini saqlashga majbur bo‘ldi. Demak asosiy muammo faqat router emas: original dense Qwen neuronlari sparse reusable primitive sifatida tashkil topmagan.

Bu taklifning savoli:

> Dense pretrained FFN funksiyasini yangi learned latent basisga qayta ifodalab, keyin faqat kichik basis subsetini ishlatish mumkinmi?

Bu pruning emas. Original neuron IDlarini saqlash shart emas.

## Asosiy graph

```text
Qwen FFN teacher
      |
      v
learned reusable basis circuits B1..Bn
      |
      +--> sparse selector / coefficients
      |
      v
global output-aware decoder
      |
      v
student FFN output
```

Maqsad teacher funksiyasini yangi koordinata sistemasida ifodalash.

## Nega bu V0.46dan keyingi to‘g‘ri qadam

V0.46 adaptive oracle natijasi shuni ko‘rsatdi: contiguous Qwen chunklarining o‘zi juda distributed. Arzonroq router oracle upper bounddan o‘ta olmaydi. Shuning uchun keyingi o‘zgaruvchi router emas, representation bo‘lishi kerak.

Synthetic Neural Engine’da independent circuit rows o‘rniga shared factorized representation routing/credit assignmentni yaxshiladi. Bu real Qwen uchun aynan o‘sha factorization ishlaydi degani emas, lekin reusable basis hypothesisni asoslaydi.

## Minimal experiment

Birdan barcha 28 layerni almashtirmang.

1. Qwen3-0.6B teacher freeze qilinadi.
2. Faqat late 4–6 FFN layer tanlanadi.
3. Har tanlangan layer uchun learned basis bank yaratiladi.
4. Teacher MLP input/output pairs bilan local distillation.
5. Keyin selected layers birgalikda end-to-end teacher-logit distillation oladi.
6. 100%, 75%, 50%, 25% active basis budgetlar o‘lchanadi.

## Representation variantlari

Birinchi phase ataylab kichik bo‘lsin:

```text
A. shared low-rank nonlinear basis
B. factorized basis + per-route coefficients
C. basis mixture + small global decoder
```

Bir vaqtning o‘zida ko‘p arxitektura variantini aralashtirmang. A ishlamasa sabab yoziladi, keyin B.

## Required controls

- original teacher;
- exact contiguous-circuit graft;
- contiguous contribution oracle;
- same active-compute compact SwiGLU baseline;
- same training-token continued-training control;
- random/magnitude basis ablation where meaningful.

## Metrics

```text
teacher CE / student CE
CE delta
logit KL
teacher top-1 agreement
selected-layer MSE
active FFN MAC/FLOP fraction
physical parameters
active parameters/token
latency
throughput
route/basis utilization
```

## Strong gate

Late-layer pilot uchun STRONG GO misoli:

```text
25% active compute
CE delta <= 0.02–0.05
high teacher top-1/KL fidelity
clearly better than contiguous oracle/compact baseline
```

Bu threshold sacred emas, lekin current V0.46 natijasidan material yaxshilanish talab qilinadi.

## Stop rule

Agar well-trained latent basis late layersning o‘zida:

- 50% active budgetda ham meaningful fidelity/compute win bermasa; yoki
- 25% active budgetda contiguous oracle’dan sezilarli ustun bo‘lmasa,

unda Qwen FFN sparsificationni vaqtincha NO-GO deb muzlatish kerak. Yana o‘nlab router variantlariga o‘tilmaydi.

## Macro-cell bilan farqi

Bu taklif **katta macro-cell** g‘oyasi emas. Basis elementlari hozircha kichik reusable function components. Macro-cell architecture alohida taklif sifatida keyin muhokama qilinadi.
