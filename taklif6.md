# Taklif 6 — validation of hidden scaling and capacity growth

Status: **keyingi ish rejasi**

## Hozirgi xulosa

V0.37 factorized global bank 20M/100M/300M all-pairs full-domain gridda
96.22%/96.25%/96.40% berdi. Active estimate uchalasida ham taxminan 1.79M.
Shu sabab typed-register + multiplicative pair + reusable factor bank hozirgi
eng yaxshi arxitektura.

500Mda esa global factorized 94.72%, factor-address 95.55% va depth-capped
global 94.81% bo‘ldi. Ularning hech biri 300M 96.40%ni tiklamadi. 500Mni
ko‘r-ko‘rona kattalashtirish to‘xtatildi.

## Keyingi tajriba

1. 20M/100M/300M factorized all-pairs uchun ikkinchi seed bilan validation.
2. Eng yaxshi 300M checkpointda hidden pair adaptationni 5k va 10k qadamda,
   visible pairlarni task-balanced sampler bilan solishtirish.
3. Hidden stage’da faqat `add → multiply` va `multiply → add` emas, barcha
   composition pairlar bo‘yicha explicit balanced mini-batch curriculumni
   o‘lchash.
4. 300M factorized parentdan 500M virtual capacityga warm-start qilib,
   yangi factor/addresslarni sekin ochish; scratch 500M bilan bir xil full-grid
   protokolda taqqoslash.

## Qabul mezonlari

- Ikkinchi seed 300M all-pairs natijani 95%dan pastga tushirmasligi;
- hidden full-grid variance 20M/100M/300M orasida tushuntiriladigan bo‘lishi;
- 10k hidden adaptation 5kdan yaxshiroq yoki kamida teng bo‘lishi;
- parent-growth 500M 300M reference’ni tiklasa, keyin 700M/1B feasibility
  screen; aks holda 300Mda qolish.

## Metodik chegara

Bu benchmark kichik modular arithmetic composition taskidir. 96%+ natija
general language yoki real-world reasoning isboti emas. Keyingi bosqichda
avval seed variance va out-of-distribution composition tasdiqlanadi.

## Foydalanilgan natijalar

- `results/V0_37_FACTORIZED_VIRTUAL_CAPACITY.md`;
- `results/V0_38_FACTOR_ADDRESS_ROUTER.md`;
- `results/V0_39_FACTOR_PAIR_BILINEAR_ROUTER.md`;
- `results/V0_40_DEPTH_CAPPED_ROUTING.md`.
