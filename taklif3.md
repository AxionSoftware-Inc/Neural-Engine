# taklif3.md — Qwen FFN -> Neural Engine exact circuit decomposition

## Maqsad

Pretrained Qwen modelidagi bilimni noldan qayta o‘qitmasdan Neural Engine falsafasiga yaqinlashtirishning eng arzon va eng toza birinchi tajribasi.

Bu bosqichda attention, embeddings, layer norms va qolgan Transformer qismlari o‘zgarmaydi. Faqat har bir Transformer FFN/SwiGLU bloki matematik jihatdan ekvivalent mayda circuit guruhlariga ajratiladi.

## Asosiy g‘oya

Qwen FFN ichidagi intermediate dimensionni bo‘laklarga ajratish mumkin. Agar barcha bo‘laklar ishlatilsa, ularning yig‘indisi original FFN natijasini qayta tiklashi kerak.

Konseptual:

```text
Original FFN
x -> [12288 hidden units] -> y

Convert:
C1 = units 0..127
C2 = units 128..255
...
C96 = last group

output = C1(x) + C2(x) + ... + C96(x)
```

Bu yerda circuitlar yangi random parametr emas. Ular original Qwen weightlarining slice’lari.

## Nega bu muhim

Agar conversion numerical tolerance ichida original logitsni saqlasa:

- pretrained bilim saqlanadi;
- modelning FFN capacitysi circuit-bank ko‘rinishiga o‘tadi;
- keyingi bosqichda router qo‘shish mumkin;
- Qwen’ni scratchdan qayta train qilish shart emasligi uchun birinchi real dalil olinadi.

## Birinchi model

Qwen3 oilasidagi eng kichik ochiq dense checkpointdan boshlash tavsiya qilinadi, masalan 0.6B-class model.

8B bilan boshlamang. Birinchi tajriba conversion correctness haqida, capability haqida emas.

## Implementatsiya

1. Original Qwen checkpointni yuklash.
2. Har bir FFN layerning up/gate/down projection weightlarini aniqlash.
3. Intermediate dimensionni fixed-size chunklarga bo‘lish: 32, 64, 128 yoki 256 neuron per circuit.
4. Har circuit original weight slice’ini aynan meros oladi.
5. Barcha circuitlar active bo‘lgan holda original FFN bilan output comparison qilish.
6. Layer-by-layer va full-model logits comparison qilish.

## Required correctness tests

Har layer uchun:

```text
max_abs_error
mean_abs_error
relative_error
cosine_similarity
```

Full model uchun:

```text
same prompt -> original logits vs converted logits
same generated tokens under deterministic decoding
perplexity delta
```

## GO / NO-GO

STRONG GO:

```text
full-model logits difference ~= floating-point numerical noise
perplexity delta negligible
```

WEAK GO:

```text
small stable numerical difference
no meaningful benchmark degradation
```

NO-GO:

```text
conversion itself causes material quality loss
```

Agar NO-GO chiqsa router/trainingga o‘tmasdan decomposition algebra va implementationni tuzatish kerak.

## Nima claim qilinmaydi

Bu bosqich sparse inference emas.

Barcha circuitlar active bo‘ladi. Maqsad faqat:

> pretrained Qwen FFN funksiyasini Neural Engine circuit representationga lossless yoki near-lossless ko‘chirish.

## Keyingi bosqich

Agar taklif3 muvaffaqiyatli bo‘lsa, `taklif4.md` bo‘yicha router yordamida active circuit sonini bosqichma-bosqich kamaytirish sinoviga o‘tiladi.
