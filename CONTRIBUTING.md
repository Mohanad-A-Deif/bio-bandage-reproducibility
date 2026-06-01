# Contributing

Contributions are welcome if they improve reproducibility, documentation, or scientific transparency.

Please keep the following rules:

1. Do not describe the reconstructed CSV files as clinical, animal, ex-vivo, or bench-top data.
2. Preserve the distinction between synthetic computational reconstruction and independent validation.
3. Keep new figures reproducible from source code.
4. Document any numerical change that affects manuscript-level results.
5. Run the reproducibility command before submitting changes:

```bash
python src/bio_bandage_reproducible_study.py --all
```
