# V0.181 Qwen contribution-space micro-group audit

Status: `REJECTED`

## Savol

Oldingi activation clustering faqat SwiGLU scalar coefficientlarini ko‘rgan.
Bu sinov neuron contribution’larini haqiqiy output-space’da cluster qiladi:

```text
contribution_j(x) = SiLU(gate_j(x)) * up_j(x) * down_proj[:, j]
```

Maqsad — har bir sparse groupning output funksiyasi coherent bo‘lishi va
top-4/8 route’da teacher FFN’ning qolgan hissasini yaxshiroq tiklashi.

## Protocol

Qwen3-0.6B, 25–26 qatlamlar, 8 copied groups, top-4 hard route (50% active),
rank-64 cross-group correction, grouped dispatch, diverse calibration
`data/qwen_calibration.txt` va mustaqil held-out `data/qwen_eval.txt`.
Asosiy smoke 100 soft + 50 hard qadamda, seed 2026da bajarildi. Har bir
neuron contribution signature 32 output coordinate’li deterministic sketch
orqali, 256 calibration tokenida balanced cluster qilindi.

## Results

| Route | Training | alpha=0 CE delta | Teacher top-1 | Local MSE (layer 25 / 26) | Gate |
|---|---:|---:|---:|---:|:---:|
| Contribution-cluster learned router | 100 + 50 hard | +0.1399 | 83.79% | 2.13 / 2.87 | FAIL |
| Contribution-cluster exact best-subset oracle | no training | +0.1779 | 83.54% | 2.77 / 4.89 | FAIL |

Oracle learned routerdan ham yomon chiqdi, chunki oracle bu partition ichidagi
eng yaxshi 4-group kombinatsiyani tanlasa ham sparse output teacher funksiyasini
yetarli tiklamadi. Bu experiment router candidate retrieval emas, group
decomposition bottleneckini ko‘rsatadi. Timing smoke’da learned variant
`1.166x` parentdan sekin bo‘ldi; contribution clustering active compute’ni
kamaytirsa ham runtime yoki quality foydasi bermadi.

## Qaror

`contribution-cluster` defaultga qabul qilinmaydi va 4-layer scale run
qilinmaydi. Output-space contribution’larni oddiy similarity bo‘yicha bir
groupga yig‘ish sparse subsetning missing signed contributions muammosini
hal qilmadi. Bu router muammosi fundamental emas degani emas, lekin ushbu
partitionda routerga qo‘shimcha training berish asosiy yo‘l emasligini
ko‘rsatadi.

Keyingi yuqori qiymatli yo‘l — har bir active micro-group full output-space’ni
qamrab oladigan signed/overlapping reconstruction yoki teacher-derived
functionally coherent basis. Held-out text gate boshidan saqlanadi; yangi
700M/1B capacity run qilinmaydi.

## Reproduction

```powershell
python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 4 `
  --partition-mode contribution-cluster --calibration-rank 64 `
  --calibration-mode cross-group --dispatch-mode grouped `
  --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --train-batches 4 --eval-batches 2 `
  --batch-size 8 --sequence-length 128 --steps 100 --hard-train-steps 50 `
  --hard-learning-rate 3e-4 --seed 2026 `
  --output results/runs/qwen_contribution_cluster_2layers_seed2026.json
```

Oracle control adds `--route-source oracle-subset --router-target subset`
and keeps `--steps 0 --hard-train-steps 0`.
