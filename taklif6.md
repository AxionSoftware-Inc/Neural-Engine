# Taklif 6 — validation of hidden scaling and capacity growth

Status: **500M/700M parent-growth ijobiy; unseen-range generalization keyingi asosiy muammo**

## Hozirgi xulosa

V0.37 factorized global bank 20M/100M/300M all-pairs full-domain gridda
96.22%/96.25%/96.40% berdi. Active estimate uchalasida ham taxminan 1.79M.
Shu sabab typed-register + multiplicative pair + reusable factor bank hozirgi
eng yaxshi arxitektura.

500Mda esa global factorized 94.72%, factor-address 95.55% va depth-capped
global 94.81% bo‘ldi. Ularning hech biri 300M 96.40%ni tiklamadi. 500Mni
ko‘r-ko‘rona kattalashtirish to‘xtatildi.

## Bajarilgan validation

1. 20M/100M/300M factorized all-pairs uchun ikkinchi seed bilan validation.
2. 20M second-seed all-pairs 96.73% berdi; seed17/18 mean 96.48%. Hidden
   seed18 adaptationda exploration 5%dan 0%ga tushirilganda full hidden
   96.38%dan 97.16%ga chiqdi. Tafsilot `results/V0_41_SEED_AND_HIDDEN_EXPLORATION.md`da.
3. 100M/300M second-seed all-pairs mos ravishda 93.80% va 95.39% berdi.
   Seed17 bilan mean 95.03%/95.89%; katta banklarda optimization variance
   kuchliroq ekanligi tasdiqlandi.
4. Eng yaxshi 300M checkpointda hidden pair adaptationni 5k va 10k qadamda,
   visible pairlarni task-balanced sampler bilan solishtirish.
5. Hidden stage’da faqat `add → multiply` va `multiply → add` emas, barcha
   composition pairlar bo‘yicha explicit balanced mini-batch curriculumni
   o‘lchash.
6. 300M factorized parentdan 500M virtual capacityga warm-start qilindi:
   scratch 500M 94.72% bo‘lgan joyda parent-growth 99.66% berdi. Hidden
   adaptation 99.32%, barcha 9 juftlik post-adaptation 99.77% bo‘ldi.
   Tafsilot `results/V0_42_PARENT_GROWTH_FACTORIZED.md`da.

## Yakuniy yangi natijalar

1. 500M parent-growth seed18 full-grid: **99.6675%**; seed17/18 mean
   **99.6622%**.
2. Combination-holdout protokoli (har pairdagi 25% operand uchligi held-out)
   300Mda 91.49%, 500M parent-growthda **99.68%** berdi.
3. 0–31da train, 32–63da OOD evaluation 300Mda 27.35%, 500Mda 28.57%
   bo‘ldi. Typed-only router 28.14% berdi, ya’ni routingni value’dan ajratish
   yetarli bo‘lmadi.
4. 500Mdan 700Mga parent-growth 3k screen’da 99.816%, clean 10k run’da
   **99.663%** berdi; active estimate 1.792M bo‘lib qoldi.

## Keyingi yo‘l

1. 700M checkpointni quality control sifatida muzlatish.
2. Unseen-range muammosi uchun representation yoki teacher-distillation
   variantini sinash; raw Qwen3 Transformer neuronlarini to‘g‘ridan-to‘g‘ri
   ko‘chirmaslik.
3. Qwen3-style teacher → Neural Engine logit/activation adapter pilotini
   kichik modelda o‘lchash.
4. Teacher/generalization natijasi ijobiy bo‘lsa, keyin 1B feasibility; salbiy
   bo‘lsa, typed-register register/circuit objective’ini qayta ko‘rib chiqish.

## Qabul mezonlari

- Ikkinchi seed 300M all-pairs natijani 95%dan pastga tushirmasligi;
- hidden full-grid variance 20M/100M/300M orasida tushuntiriladigan bo‘lishi;
- 10k hidden adaptation 5kdan yaxshiroq yoki kamida teng bo‘lishi;
- parent-growth 500M va 700M supported/structured compositionda ijobiy chiqdi;
  unseen value-range muammosi sabab 1Bga shoshilmaslik kerak.

## Metodik chegara

Bu benchmark kichik modular arithmetic composition taskidir. 96%+ natija
general language yoki real-world reasoning isboti emas. Keyingi bosqichda
avval seed variance va out-of-distribution composition tasdiqlanadi.

## Foydalanilgan natijalar

- `results/V0_37_FACTORIZED_VIRTUAL_CAPACITY.md`;
- `results/V0_38_FACTOR_ADDRESS_ROUTER.md`;
- `results/V0_39_FACTOR_PAIR_BILINEAR_ROUTER.md`;
- `results/V0_40_DEPTH_CAPPED_ROUTING.md`.
- `results/V0_41_SEED_AND_HIDDEN_EXPLORATION.md`;
- `results/V0_42_PARENT_GROWTH_FACTORIZED.md`.
- `results/V0_43_OOD_AND_700M_GROWTH.md`.
