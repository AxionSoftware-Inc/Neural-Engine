# Neural-Engine: qayerda xato qildik va buyog‘iga nima qilamiz?

Sana: 2026-09-06. Ko‘rilgan holat: `exp/scale-invariant-routing`, commit `9e565ee` — “fix full active route scaling”.

Bu mustaqil tahlil va tavsiya, bajarilgan tuzatishlar hisoboti emas. Manbalar: loyiha kodi, Git tarixi, tajriba hisobotlari va saqlangan JSON natijalari. Yangi training, inference yoki GPU testi ishga tushirilmadi. Asosiy agent ishiga aralashilmadi; mavjud kod va natijalar o‘zgartirilmadi.

## Qisqa hukm

Eng katta xato — ayrim muvaffaqiyatsiz tajribalardan arxitektura haqida xulosa chiqarishdan oldin, training va inference aynan bir xil funksiyani hisoblashini hamda taqqoslanayotgan runlar haqiqatan teng sharoitda ekanini yetarli tekshirmaganmiz.

Hozirgi dalil “sparse arxitektura ishlamaydi” ham emas, “faqat bitta bugni tuzatsak hammasi ishlaydi” ham emas. Dalil shuki: benchmarkda ikkita muhim amalga oshirish muammosi bor; ayrim nazorat taqqoslashlari aralash omillar bilan o‘tkazilgan. Ularning sifatga qo‘shgan ulushini tuzatilgan, teng sharoitdagi qayta sinov aniqlaydi.

Mening tanlovim: avval tajriba ishonchliligini tiklash; keyin mavjud rank-64 correction bilan, haqiqiy hard-route rejimida, ikki qatlamlik bloklarni birgalikda distillatsiya qilishni sinash. Hozir yangi katta arxitektura yoki 700M/1B masshtabga o‘tmaslik.

## 1. Koddan aniq ko‘ringan muammolar

### 1.1. Hard training va grouped inference masshtabi bir xil emas

`TransferredRoutedQwenChild` hard training vaqtida token-loop yo‘lidan o‘tadi. Bu yo‘l natijani qattiq yozilgan `E/K` koeffitsiyentiga ko‘paytiradi. Grouped inference esa sozlamadan olingan `hard_route_scale`ni ishlatadi. [Dispatch tanlovi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:685), [token-loop masshtabi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:715), [grouped masshtabi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:591).

V0.167 ning muvaffaqiyatli to‘rt qatlamli runida E=8, K=4 va explicit scale=4. Demak, hard training bazaviy chiqishni 2 ga, inference esa 4 ga ko‘paytirgan.

Bu oddiy sonli aniqlik farqi emas. Subset-router tanlangan to‘rtta guruhga teng 1/4 vazn berganda, U tanlangan guruhlarning asl chiqishlari yig‘indisi bo‘lsin:

- Hard trainingdagi bazaviy chiqish: (8/4) × (U/4) = U/2.
- Grouped inferencedagi bazaviy chiqish: 4 × (U/4) = U.

Correction qismi ham mavjud; bu hisob butun model chiqishi doim ikki baravar degani emas. Ammo correction trainingda boshqa bazaviy funksiyaning xatosini tuzatishga o‘rgatiladi. Uning wrapperidagi masshtab [alohida shu konfiguratsiyadan olinadi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1033).

Oxirgi V0.173 tuzatishi full-active grouped yo‘l uchun foydali: K=E bo‘lganda default scale endi E. Lekin token-loopdagi qattiq yozilgan E/K qolgan. Shuning uchun tuzatish barcha bajarilish yo‘llarini hali tenglashtirmagan.

Yana bir shart: softmax vaznlari teng bo‘lmasa, scale=E ning o‘zi ham barcha guruhlar yig‘indisini kafolatlamaydi. Exact full-active nazorati routerdan mustaqil to‘g‘ridan-to‘g‘ri yig‘indini yoki aniq teng vazn shartini tekshirishi kerak.

Xulosa: bu statik koddan tasdiqlangan nomuvofiqlik. Uning qaysi runni qanchaga yomonlashtirgani hali o‘lchanmagan. Ijobiy natijalar ham shu ogohlantirish bilan qayta tekshirilishi kerak.

### 1.2. Refinement “muzlatilgan” Qwen vaznlarini ochadi, inference esa eski nusxalarini o‘qiydi

Ko‘chirilgan Qwen FFN slice vaznlari dastlab [muzlatiladi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:358). Grouped bajarish uchun ulardan alohida stacked bufferlar [konstruktor paytida yaratiladi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:479).

Ammo joint refinement barcha `child.parameters()`ni optimizerga beradi va hammasida gradientni yoqadi. Layerwise refinement ham shunday qiladi. Bunga copied gate/up/down vaznlari ham, child ichidagi router parametrlari ham kiradi — faqat correction emas. [Joint](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1612), [layerwise](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1699).

Natijada:

- Hard training original expert modullaridagi jonli vaznlarni o‘qiydi.
- Refinement shu parametrlarni yangilashga ruxsat beradi.
- Grouped inference konstruktor vaqtida yig‘ilgan cached nusxalarni o‘qiydi.
- Ko‘rilgan kodda bu nusxalarni yangilangan expert vaznlari bilan qayta sinxronlash yo‘li topilmadi.

Demak, refinementdan keyin optimizer o‘rgatgan funksiya bilan baholanayotgan funksiya ajralishi mumkin. Bu faqat “KL objective yomon ekan” degan izoh bilan qoplanmaydi.

Birinchi tuzatilgan protokolda copied Qwen vaznlari va router chindan muzlatilsin; faqat oldindan sanab qo‘yilgan correction parametrlari o‘rgatilsin. Agar keyinchalik copied vaznlarni ham o‘rgatmoqchi bo‘lsak, bu alohida tajriba bo‘lsin: boshqa trainable budget va cache izchilligi bilan.

## 2. Tajribalarni talqin qilishda xatolar

### 2.1. “Bir xil sharoitdagi” ayrim taqqoslashlar aslida bir xil emas

Quyidagi qiymatlar hisobotdagi ta’rifdan emas, run JSONlaridan olingan. Barchasi seed 2026, alpha=0. “Default 2” — JSONda scale null, E=8/K=4 uchun kodning effektiv qiymati 2 degani.

| Tajriba | Qatlamlar | Inference scale | Hard LR | +CE |
|---|---:|---:|---:|---:|
| V0.167 frozen subset-router nazorati [R1] | 4 | explicit 4 | 1e-4 | +0.0095 |
| V0.169 layerwise refinement [R2] | 4 | default 2 | 1e-4 | +0.4703 |
| V0.170 output contract [R3] | 4 | default 2 | 1e-4 | +0.1549 |
| V0.168 uchun sakkiz qatlamli nazorat [R4] | 8 | explicit 4 | 1e-5 | +0.5685 |
| V0.168 joint refinement [R5] | 8 | default 2 | 1e-5 | +0.9811 |

Shu sabab +0.0095 → +0.4703 farqini faqat layerwise refinementga, +0.5685 → +0.9811 farqini faqat joint objectivega yuklash mumkin emas. Masshtab ham o‘zgargan; refinementda yuqoridagi vazn/cache muammosi ham bor.

Child parametr soni ham ayrim juftlarda 10,758,222 dan 10,625,990 ga o‘zgargan: modul tarkibi bir xil qolmagan. Bir xil global seed bir xil boshlang‘ich vaznlar degani emas — oldin qancha modul yaratilgani ham random generator holatiga ta’sir qiladi.

To‘g‘ri A/B: bitta saqlangan boshlang‘ich checkpointdan ikki nusxa, bir xil tokenlar va tartib, bir xil effektiv sozlamalar; faqat tekshirilayotgan bitta omil o‘zgarsin.

### 2.2. Refinementdan oldingi MSE keyingi oqibat deb yozilgan

[V0.169 hisoboti](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/V0_169_QWEN_LAYERWISE_TASK_REFINEMENT_AUDIT.md) birinchi qatlam MSE=135.04 qiymatini refinementdan keyingi buzilish sifatida talqin qiladi.

Lekin JSONga yoziladigan `child_local_eval_mse` [avval hisoblanadi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1987); layerwise/joint refinement esa [undan keyin chaqiriladi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:2003). Ushbu qiymatlar refinementdan keyin qayta hisoblanmaydi.

Demak, MSE=135.04 boshlang‘ich child tayyorlash bosqichidayoq mavjud bo‘lgan. Uni keyingi refinement keltirib chiqargan deb bo‘lmaydi. Endi har bir metrika bosqichi aniq nomlansin: initial, post-local-train, pre-refine, post-refine, final-cascade.

### 2.3. V0.173 hali toza “depth cascade” nazorati emas

[V0.173](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/V0_173_QWEN_FULL_ACTIVE_SCALE_FIX_AUDIT.md) va [R6] natijasida 19–22 qatlamlar full-active exact, 23–26 esa sparse. Dastlabki to‘rtining local MSEsi taxminan 1e-12 — bu foydali exactness dalili.

Lekin bu konfiguratsiyada ham, odatiy to‘rt qatlamli almashtirishda ham taxminiy ishlayotgan qatlamlar soni to‘rtta. Qolgan Transformer qatlamlari avvaldan modelda mavjud. Exact wrapperlar qo‘shilishi o‘z-o‘zidan to‘rtta qo‘shimcha approximation qatlami hosil qilmaydi.

Bu run suffixni qayta o‘rgatgan; hard LR 1e-5, muvaffaqiyatli V0.167 nazoratida esa 1e-4. Modul yaratish tartibi ham boshqa. Shuning uchun +0.7671 “exact prefix bo‘lsa ham depthning o‘zi yiqitadi” degan hukm uchun toza dalil emas.

Kerakli nazorat: aynan bir xil muzlatilgan 23–26 suffix vaznlarini olib, faqat 19–22 dagi parent FFNlarni exact nusxalar bilan almashtirish; hech narsani qayta o‘rgatmaslik. Chiqishlar numerical tolerance ichida teng bo‘lishi kerak. Farq bo‘lsa, avval boundary hidden state, route tanlovi va rejim/cache farqlari tekshirilsin. Kichik sonli farq top-k qarorini almashtirsa, bu ham alohida o‘lchanadigan routing beqarorligi; umumiy depth isboti emas.

### 2.4. “Sparse prefix ma’lumotida o‘rgataylik”ning bir qismi allaqachon bor

Hozirgi sikl childni modelga o‘rnatadi, keyingi qatlam inputlarini esa shundan keyin capture qiladi. Demak, keyingi child avvalgi sparse prefix ishlab chiqargan inputlarda kalibrlanadi. [Capture](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1870), [childni o‘rnatish](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1997).

Yangi tajriba shunchaki shu taklifni qaytarmasligi kerak. Haqiqiy yangi omil — qo‘shni qatlamlarning umumiy chiqishini o‘rgatish va prefix parametrlari keyin o‘zgarsa, downstream kalibratsiya taqsimotini qayta yangilash.

## 3. Baholash va “tejamkorlik” haqidagi cheklovlar

Hozirgi [calibration matni](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/data/qwen_calibration.txt) taxminan 15 ming, [eval matni](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/data/qwen_eval.txt) 6.8 ming belgidan iborat. Ikkalasi ham qisqa, mavzulari yaqin inglizcha texnik matnlar. Bu broad language-model sifatiga yetarli qamrov emas.

Token oqimi yetmasa [matn takrorlanadi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_parent_transplant.py:66). Shu sabab 8192 training / 4096 evaluation pozitsiyasi shuncha mustaqil yangi token yoki hujjat degani emas. Ko‘p variant tanlashda ishlatilgan shu evalni bundan keyin development to‘plami deb qabul qilish kerak. Bu evalda gradient hisoblangan degan ayblov emas; bir testni takror-takror tanlov uchun ishlatish yakuniy xulosaning mustaqilligini kamaytiradi.

Hozirgi gate asosan +CE ≤ 0.05 va finite CE. Teacher top-1 o‘xshashligi gate sharti emas. Shuning uchun +0.0095 va taxminan 82% top-1 bir-biriga zid emas: biri matn ehtimoli, boshqasi teacher tanlovlariga moslik. “CE saqlandi”ni “funksiya to‘liq saqlandi”ga aylantirmaslik kerak. [Gate](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:2176).

Teacher logitlari [float16 saqlanib](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1819), qayta float32ga olinishi exact reference tekshiruvini ham chalkashtiradi: parent bilan parent nazoratida top-1 taxminan 99.6%. Identity testi uchun ikkala tomonda bir xil aniqlikdagi reference ishlatilsin; bu yerda 100%dan farq sparse modelning xatosi emas.

Joint KL `batchmean` bilan [B,T,V] tensorida hisoblangan. U batch B bo‘yicha bo‘linadi, B×T bo‘yicha emas; T=128 bo‘lsa, token-o‘rtacha KLdan 128 marta katta son chiqadi. Bu o‘z-o‘zidan noto‘g‘ri objective emas, lekin “loss 70 bo‘ldi, demak falokat” deyish yetarli emas. Valid token bo‘yicha normallash va gradient masshtabini alohida qayd etish kerak. [Mahalliy hisob](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:1633), [PyTorch rasmiy ta’rifi](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html).

K=4/E=8 — almashtirilgan FFN body guruhlarining 50%i tanlandi degani. Bu butun model FLOPi, xotirasi yoki vaqti 50% kamaydi degani emas:

- Attention va boshqa qatlamlar saqlangan; correction/router ham ish bajaradi.
- Grouped yo‘l guruhlarni eng band guruh token sonigacha [padding qiladi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:541). Foydali juftlar T×K, bajarilgan padded juftlar E×max_count. Foydalanish koeffitsiyenti: (T×K)/(E×max_count).
- Barcha token bir xil guruhlarni tanlasa, max_count=T bo‘lishi mumkin; grouped matmulning body ishi dense darajasiga yetadi.
- Hisobotlardagi taxminan 2.14× child storage va 1.32× end-to-end vaqt — hozircha deployment yutug‘i emas. [V0.167](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/V0_167_QWEN_COUPLED_SUBSET_ROUTER_AUDIT.md).

## 4. Nimalarni saqlab qolish kerak?

Ijobiy izlanishlarni bekor qilish kerak emas. Cross-group correction va exact best-subset nazoratlari foydali signal bergan. Lekin ularning xulosasini mavjud korpus, operator va protokol chegarasida ushlash kerak.

[V0.161](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/V0_161_QWEN_CROSS_GROUP_MIXING_SCALE.md) to‘rt qatlamda ikki seed uchun +0.0181 va +0.0155 ko‘rsatgan. V0.167 esa +0.0095 va +0.0454. Demak, V0.167 ni har jihatdan mutlaq eng yaxshi deb atamaymiz: bir seedda yaxshiroq, boshqasida yomonroq. Ikkala konfiguratsiya ham qayta tiklanadigan reference sifatida qolsin.

Oddiy dot/energy “oracle” muvaffaqiyatsizligi barcha sparse subsetlar yomonligini isbotlamagan; V0.167 dagi 70 variantli exact subset nazorati buni amalda ko‘rsatgan. Xuddi shunday, hozirgi joint/layerwise runlar ham bu o‘qitish oilalarini butunlay rad etishga yetmaydi.

“Eng yaxshi konfiguratsiya” va “saqlangan eng yaxshi checkpoint”ni ajratish kerak. Ko‘rilgan benchmarkning yakunida JSON yoziladi, lekin model vaznlarini saqlash ko‘rinmaydi. Kelajakdagi nazoratlar uchun o‘sha vaznlarning o‘zi, partition/subset tartibi va qayta yuklash tengligi zarur. [Saqlash qismi](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_multi_layer_transplant.py:2185).

## 5. Men taklif qiladigan ish tartibi

Quyidagilar keyingi implementatsiya agenti uchun reja. Bu hujjat ularni hozir bajarish topshirig‘i yoki mavjud ishni to‘xtatish talabi emas. Asosiy agent bilan kelishilgan bo‘sh vaqtda, alohida tajriba identifikatorlari ostida bajarilsin.

### Bosqich A — yangi trainingdan oldin hisoblash tengligini isbotlash

1. E=8, K=4/6/8 uchun token-loop, hard-training forward va grouped inference bir xil input, vazn, tanlangan guruh va route vaznlarida taqqoslansin. Default va explicit scale holatlari alohida qamralsin. Soft barcha-guruh yo‘li sparse hard yo‘lga teng bo‘lishi talab qilinmaydi.
2. Correction o‘chirilgan va yoqilgan holatlar tekshirilsin. Full-active exact nazorat teng vazn va to‘g‘ridan-to‘g‘ri yig‘indi shartini alohida tekshirsin.
3. Bir optimizer qadamidan keyin ayni tenglik testi takrorlansin. Faqat ruxsat etilgan correction parametrlari o‘zgargani; copied FFN va muzlatilgan router hash/qiymatlari o‘zgarmagani tekshirilsin.
4. Saqlash–qayta yuklashdan keyin ham chiqishlar, partition va route tanlovi tiklansin. Non-persistent cached bufferlar uchun bu ayniqsa muhim.
5. Yuqoridagi exact-prefix A/B bir xil sparse suffix bilan, trainingsiz o‘tkazilsin. Saqlangan mos checkpoint bo‘lmasa, avval bittasi tenglikdan o‘tgan yo‘lda tayyorlanib muzlatilsin.

O‘tish sharti: farqlar dtype va hisoblash tartibining odatiy numerical shovqinidan oshmasin. FP32 uchun boshlang‘ich tolerance rtol=1e-5, atol=1e-6 bo‘lishi mumkin, lekin activation masshtabi va parentning o‘z numerical nazorati bilan tekshirilsin; katta xatoni yashirish uchun bo‘shatilmasin. Route replay bilan algebraik xato, erkin routing bilan top-k chegarasiga sezgirlik ajratilsin.

Bu bosqich o‘tmasa, sifat bo‘yicha yangi arxitektura xulosasi chiqarilmasin.

### Bosqich B — kichik, qayta tiklanadigan reference

Avval ikki qatlam, so‘ng to‘rt qatlam. V0.161 va V0.167 yondashuvlari tuzatilgan operator bilan qayta o‘tkazilsin. Eski raqamni aynan qaytarish kafolatlanmaydi — training funksiyasi tuzatilmoqda. Oldingi runlar tarix sifatida saqlansin, yangilari “corrected protocol” deb ajratilsin.

Har run uchun quyidagilar saqlansin:

- Kod commit/hash, to‘liq effektiv argumentlar, model/tokenizer revisioni, dtype va aniq token IDlari.
- Boshlang‘ich/final checkpoint; qatlamlar bo‘yicha alohida random seed yoki saqlangan bir xil boshlang‘ich vaznlar.
- Soft va hard qadamlarning haqiqiy soni. Masalan, joriy [train_child mantiqida](C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/benchmark_qwen_two_layer_transplant.py:438) steps=300 va hard_steps=100 — jami 300, ya’ni 200 soft + 100 hard; 300+100 emas.
- Trainable parametrlar ro‘yxati, scale, barcha metrikalarning o‘lchangan bosqichi.
- Token-batch kesimida o‘sha batchning teacher CEsi bilan juft farq; global teacher o‘rtachasiga nisbatan “per-batch delta” bilan aralashtirmaslik.

Birinchi ekran uchun seed 2026 va 2027; istiqbolli bitta nomzod uchun uchinchi seed 2028. Bitta omadli seed bilan qaror chiqarmaslik.

Data uchun boshlang‘ich resurs taklifi: takrorlanmagan 32–64 ming calibration token, 16 ming development token, alohida 32 ming final-test token. Bu kafolatlangan yetarli hajm emas, kichik loyiha uchun boshlang‘ich budjet. Hujjat/manba bo‘yicha ajratilsin; turli nasr, kod va hisoblash matnlari, maqsadga mos bo‘lsa o‘zbekcha segment ham bo‘lsin. Final test faqat usul muzlatilgandan keyin ochilsin.

### Bosqich C — xatoning manbasini ajratish

Yangi modul qo‘shishdan oldin uchta savolga javob olish kerak:

1. Shu decomposition ichida yaxshi subset bormi? Bir xil corrected operator va correction vaznlarida exact best-subset bilan learned routerni solishtirish. E=8/K=4 da 70 variant kichik diagnostik budjetda tekshiriladi. Oracle deployment emas va local MSE bo‘yicha tanlangan subset global CE optimumi degani emas.
2. Xato prefix taqsimotidan kuchayadimi? Bir qatlamni teacher-prefix inputida va haqiqiy sparse-prefix inputida alohida baholash; relative MSE, cosine, RMS, route almashishi va tanlov chegarasi o‘lchansin.
3. Qaysi joy sezgir? Cascade ichida bittadan childni parentga qaytarib, hech qanday retraining qilmasdan yakuniy +CE o‘zgarishini o‘lchash. Bu yakka qatlamning causal hissasini to‘liq additiv deb bermaydi, lekin keyingi nishonni tanlashga yordam beradi.

Router uchun faqat class accuracy yetmaydi. Uning tanlagan subseti oracle tanlovidan qanchalik ko‘p rekonstruksiya xatosi qilgani — subset regret — ham hisoblanishi kerak.

### Bosqich D — asosiy yangi tajriba: hard-route blok distillatsiyasi

Agar yuqoridagi nazoratlar o‘tsa, men birinchi bo‘lib shu yo‘lni sinardim:

- E=8/K=4 va mavjud rank-64 correction saqlanadi. Copied Qwen FFNlar ham, avval o‘rgatilgan subset-router ham muzlatiladi. Yangi quvvat qo‘shilmaydi.
- Training va inference bir xil hard operatorga ega bo‘ladi. Teng subset vaznlari bilan tanlangan asl chiqishlar yig‘indisini saqlamoqchi bo‘lsak scale=K; E/Kni avtomatik “to‘g‘ri/unbiased” deb qabul qilmaymiz. Dastlabki K=4 nazoratida explicit scale=4 har ikki yo‘lda qo‘llanadi.
- Bir xil checkpointdan A va B olinadi. A: hozirgi local hard distillation. B: local xatoni saqlagan holda qo‘shni ikki qatlam blokining chiqish xatosi ham kamaytiriladi.
- Teacher va student bloklari bir xil boshlang‘ich hidden state, position/attention sharoitidan boshlaydi. Blok ichidagi normalizatsiya, residual va attention konteksti saqlanadi; o‘zgartiriladigan parametrlar faqat correction.
- Avval 25–26, keyin 23–26 qatlamlar. To‘rt qatlamdagi barqaror natijadan keyin 22 ni, keyin yana bittadan qo‘shish. To‘g‘ridan-to‘g‘ri sakkiz qatlamga sakramaslik.
- Birinchi A/Bda final-logit KL, yangi router, yangi rank va boshqa decomposition birdan kiritilmaydi. Kerak bo‘lsa kichik token-normalized KL keyingi alohida ablation bo‘ladi.
- Prefix correction o‘zgarsa, downstream inputlar qayta capture qilinadi yoki joriy prefixdan olinadi. Teacher dense hisoblari faqat training/diagnostikada; sparse inference uchun yashirin dense yordam bo‘lmaydi.

Nega shu yo‘l? Mavjud correction foydali signal bergan, lekin faqat har FFNning local chiqishini yaqinlashtirish blokning yakuniy hidden state yo‘nalishini yetarli saqlamasligi mumkin. Ikki qatlamli objective aynan shu taxminni kamroq yangi omil bilan tekshiradi.

Bu kafolat emas. A/B bir xil qadam/token budjetida, masalan dastlab 100 hard qadamda bajarilsin. Qadamni 300 gacha uzaytirish mezoni oldindan belgilanib, ikkala tomonga bir xil qo‘llansin. Blok objective qo‘shimcha hisoblash talab qilishi mumkin; training vaqti va xotirasi ham qayd etilsin. Local va blok xatolari target energiyasiga nisbatan normallansin; ularning nisbiy vazni development sinovidan oldin belgilansin. Tuzatilgan hard-only, zero-start correction ham soft→hard protokoliga qarshi keyingi kichik nazorat bo‘lishi mumkin; birinchi A/Bga aralashtirilmasin.

### Bosqich E — faqat diagnostika talab qilsa yo‘nalishni o‘zgartirish

Agar oracle yaxshi, learned router yomon bo‘lsa: avval router ustida ishlash. Yagona qattiq label o‘rniga bir nechta yaxshi subsetning xato narxini hisobga oluvchi supervisionni sinash; hammasi bir xil inference budgetda. E katta bo‘lganda barcha subsetlarni sanash kombinatorik kattalashadi, 70-class usulni cheksiz kengaytirmaslik.

Agar corrected oracle ham yomon bo‘lsa: shunda decomposition yoki omitted residualni ifodalash yo‘lini qayta ko‘rish. Avval sodda nazorat — tokenlarga bog‘liq bo‘lmagan 50% fixed neuron/group tanlovi va reconstruction bilan moslashtirilgan output projection. Dynamic routing haqiqatan shu sodda nazoratdan foyda beradimi?

Tashqi metodik nazorat sifatida [SparseGPT](https://arxiv.org/abs/2301.00774) yoki [Wanda](https://arxiv.org/abs/2306.11695) bilan teacher-activation asosidagi pruning/reconstruction yondashuvini solishtirish mumkin. Lekin ular asosan weight sparsity haqida; 50% weight zero bilan bizdagi 50% active group bir xil hisoblash budjeti emas. Qwen3 mosligi va mazkur GPUdagi real tezlik alohida tekshiriladi. Ular darhol asosiy yo‘l emas, keyingi nazorat variantlari.

Agar fixed subset yaxshi, dynamic yomon bo‘lsa — switching/router/dispatch muammosiga e’tibor. Agar ikkalasi ham yomon bo‘lsa — representation va berilgan faol quvvat chekloviga e’tibor. Bu hali barcha sparse arxitekturalar uchun imkonsizlik isboti emas.

## 6. O‘tish va to‘xtash mezonlari

Sifat uchun tarixiy +CE ≤ 0.05 chegarasini reference sifatida saqlash mumkin. Lekin yangi qaror oldindan belgilangan ko‘p domenli development to‘plamida, barcha seedlar va juft farqlar bilan qabul qilinsin. Keyingi kengroq tasdiqda hujjatlar bo‘yicha bootstrap ishonch oralig‘i ham berilsin; takroriy tokenlarni mustaqil namunalar deb sanamaslik kerak.

Agar maqsad teacher funksiyasiga yuqori moslik bo‘lsa, top-1 va task-level javob mosligi alohida talab qilinsin. Masalan, ≥90% top-1 qo‘shimcha tadqiqot maqsadi bo‘lishi mumkin, lekin bu yangi, qat’iyroq mezon: oldingi 82% natijani retrospektiv ravishda “eski gate fail”ga aylantirmaydi. Top-1ning o‘zi ham umumiy LM sifati emas.

Tezlik uchun faqat sparse child ichki foizi emas, end-to-end prefill va KV-cache bilan batch-1 decode alohida o‘lchansin. Useful/padded FLOP, correction/router xarajati, peak VRAM va real saqlanadigan baytlar ajratilsin. Tezlik da’vosi uchun sifatga teng sharoitdagi barqaror latency/throughput yutug‘i kerak.

To‘xtash nuqtalari:

- Hisoblash tengligi buzilsa — yangi sifat runlari emas, correctness tekshiruvi.
- Tuzatilgan ikki/to‘rt qatlamli reference qayta olinmasa — sakkiz qatlamga kengaymaslik.
- Blok objective bir xil budjetda ikki seedda barqaror foyda bermasa — ko‘r-ko‘rona LR/rank sweep emas, C bosqich diagnostikasiga qaytish.
- Sifat o‘tsa-yu padded ijro real tejash bermasa — avval ijro/xotira masalasini hal qilish, keyin kattalashtirish.
- Kengroq final test muvaffaqiyatsiz bo‘lsa — natijani yashirmaslik; shu testdan keyingi tuningda foydalanilsa, endi uni dev deb hisoblash.

## 7. Hozir nimalarni qilmaslik kerak?

700M/1B/8B ga o‘tish; yangi “katta g‘oya” bilan barcha omillarni bir vaqtda almashtirish; bir xil evalda cheksiz variant tanlash; bitta seedning eng yaxshi raqamini endpoint qilish; corrected nazoratsiz joint/layerwise oilasini butunlay tashlash — hozir foydasi past.

Shuningdek, Qwen ichidagi FFN almashtirishni attention-free yangi til modeli deb atamaslik kerak. Native Neural-Engine yo‘nalishi va Qwen ichida sparse hisoblashni o‘rganish — ikki boshqa dalil zanjiri. Sintetik topshiriqdagi muvaffaqiyat real matn modeliga avtomatik ko‘chmaydi; virtual/factorized parametr soni ham jismoniy vazn va faol hisoblash soni bilan bir emas.

Mening yakuniy taklifim: birinchi navbatda scale + freeze/cache + matched checkpoint muammolarini tekshirib tuzatish; so‘ng ikki/to‘rt qatlamli corrected reference; undan keyin hard-route ikki qatlamlik blok distillatsiyasi. Faqat shu tartib muvaffaqiyatsizlikning haqiqiy sababini yangi arxitektura zaruratidan ajratib beradi.

## Run manbalari

[R1]: C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_seed2026.json
[R2]: C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_layerwise20_lr1e-5_seed2026.json
[R3]: C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_outputcontract_seed2026.json
[R4]: C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/runs/qwen_multi_layer_crossgroup_8layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-5_seed2026.json
[R5]: C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/runs/qwen_multi_layer_crossgroup_8layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-5_joint50_lr1e-5_seed2026.json
[R6]: C:/Users/shaxz/OneDrive/Dokumenty/Neural-Engine/results/runs/qwen_multi_layer_crossgroup_8layers_e8_schedule8844_rank0x4_rank64x4_scale8444_subsetrouter100_seed2026.json

Matndagi R1–R6 havolalar tegishli xom natija fayllariga olib boradi. Bu hujjatdagi yangi sababiy izohlar yuqorida alohida ko‘rsatilgan statik dalillarga tayangan; tavsiya qilingan nazoratlar hali bajarilmagan.
