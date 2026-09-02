# taklif18.md — Hardware-native grouped sparse execution and routing locality

Status: **systems proposal; required before claiming real sparse speedups**

## Muammo

Current Qwen sparse pilotsda hard sparse path ba’zan exact dense pathdan ham sekin chiqdi. Sabab architecture emasligi mumkin: generic per-token gather/einsum va dynamic dispatch overhead GPUni yomon ishlatadi.

Shuning uchun `active params` kamayishi avtomatik ravishda wall-clock speedup degani emas.

## Maqsad

Neural Engine sparse computationini GPU/NPU/CPU memory hierarchyga mos qilib, selected circuitsni random mayda gatherlar emas, **contiguous grouped pages** sifatida bajarish.

## Asosiy graph

```text
router
  ↓
small route IDs
  ↓
page/family grouping
  ↓
prefetch hot pages
  ↓
fused batched circuit kernel
  ↓
register/state update
```

## Design principles

1. Circuit/factor rows physically contiguous pagesga joylanadi.
2. Router addresslari locality-aware bo‘lishi kerak.
3. Bir batch/token guruhidagi bir xil yoki yaqin circuit IDs grouped qilinadi.
4. Gather + low-rank matmul + activation + up projection imkon qadar fused qilinadi.
5. Hot working set VRAM/cacheda, cold bank CPU RAMda qolishi mumkin.
6. CPU->GPU transfer faqat cache missda; predictive prefetch sinov qilinadi.
7. NPU target uchun fixed page sizes va bounded program templates saqlanadi.

## Minimal GPU experiment

Qwen/NE research correctnessdan alohida microbenchmark:

```text
A. current generic einsum/gather
B. grouped-by-circuit execution
C. contiguous page execution
D. fused/custom kernel if needed
```

Bir xil selected weights va bir xil output bilan solishtirish.

## Required metrics

```text
kernel latency
tokens/s or samples/s
effective memory bandwidth
H2D bytes/token
cache hit rate
router overhead
selected-weight bytes/token
GPU utilization
VRAM
CPU RAM
batch-size sensitivity
```

## Correctness gate

Optimized runtime outputi reference implementation bilan numerical tolerance ichida bir xil bo‘lishi shart.

## Speed gate

Sparse route faqat FLOP estimate bilan GO bo‘lmaydi.

STRONG GO misoli:

```text
>=4x lower active FFN/circuit compute
AND
material wall-clock latency/throughput improvement
AND
router+dispatch overhead small relative to compute
```

Agar active compute kamayib, runtime yomonlashsa architecture-quality natija saqlanadi, lekin systems claim NO-GO bo‘ladi.

## RAM-backed large bank experiment

Current factorized/virtual bank yoki keyingi real LLM student bilan:

```text
cold factors/circuits in system RAM
hot page cache in VRAM
```

O‘lchanadi:

- random route trace;
- real learned route trace;
- hit/miss distribution;
- PCIe traffic;
- prefetch accuracy;
- steady-state tok/s.

## Nega bu alohida taklif

Research architecture va runtime implementationni aralashtirmaslik kerak. V0.46dagi sparse hard pathning sekinligi `sparse idea false` degani emas; generic dispatch implementationning xarajati bo‘lishi mumkin. Shu bilan birga efficient kernel bo‘lmasdan `NE 4x/10x faster` claim qilish ham mumkin emas.

## Macro-cell bilan aloqa

Macro-cell keyinchalik tanlansa, aksincha hardware uchun foydali bo‘lishi mumkin: kamroq, kattaroq contiguous computation blocks random minglab micro-gatherlarga qaraganda osonroq batching/fusion beradi. Ammo bu proposal macro-cell arxitekturasini belgilamaydi.
