# دليل رفع المستودع على GitHub

## الاسم المقترح للمستودع

`bio-bandage-reproducibility`

## الوصف المقترح

Reproducible synthetic computational package for an in-silico diclofenac cryocompression bandage study.

## قبل الرفع

عدّل هذه القيم في الملفات:

- استبدل `Mohanad-A-Deif/bio-bandage-reproducibility` باسم حسابك واسم المستودع في `README.md` و `CITATION.cff`.
- راجع أسماء المؤلفين في `CITATION.cff`.
- راجع سياسة الترخيص في `LICENSE` و `DATA_LICENSE.md`.

## أوامر الرفع

افتح Terminal داخل مجلد المستودع ثم نفّذ:

```bash
git init
git add .
git commit -m "Initial reproducibility package"
git branch -M main
git remote add origin https://github.com/Mohanad-A-Deif/bio-bandage-reproducibility.git
git push -u origin main
```

## بعد الرفع

1. افتح صفحة المستودع على GitHub.
2. تأكد أن README يظهر بشكل صحيح.
3. افتح تبويب Actions وتأكد أن reproducibility workflow يعمل.
4. أنشئ Release باسم `v1.0.0`.
5. إذا أردت DOI، اربط المستودع بـ Zenodo ثم أنشئ Release جديد.
