# AME(8,4) — PC'de tekrar ve bağımsız kontrol

## Çalıştırma

`calistir.bat`'a çift tıkla. (Ya da Anaconda Prompt'ta bu klasöre gelip `calistir.bat` yaz.)

1. Önce doğrulama: AME(6,4) için **1552** sonucu çıkmalı. Ekranda `TOTAL ... 1552` satırını görürsün.
2. Sonra 10 küçük pencere açılır, her biri bir sınıfı tarar:
   - **Ana tarama** (`fast_search.py`): 10–15 dakika. Sonuç `pc_class<N>.log`'da.
     Beklenen: `rows at each level [..., 0, 0]  solutions 0`.
   - **Bağımsız kontrol** (`subspace_check.py all`): Bu sınıfın bütün 2. satır adaylarını, normalizasyon ve
     Schur mantığını hiç kullanmayan bir yöntemle tek tek tamamlamaya çalışır. 1–3 saat sürer.
     Sonuç `pc_subspace_class<N>.log`'da. Beklenen: `completions found: 0`.

Pencereler işi bitince kendiliğinden kapanır. Bilgisayar uyku moduna geçmesin.

## Bu ne kanıtlıyor, ne kanıtlamıyor

- Aynı kodu farklı bir makinede çalıştırmak, ortam ve donanım kaynaklı hataları eler.
- Bağımsız kontrol, 3. ve 4. satırların aranmasını başka bir algoritmayla tamamen tekrar eder.
  2. satır listesi iki yöntemde ortak (o kısım daha önce saf Python ile dilimler üzerinde karşılaştırıldı).
- Kapsam yine sadece "her ququart = 2 qubit" tipi stabilizer durumları.
