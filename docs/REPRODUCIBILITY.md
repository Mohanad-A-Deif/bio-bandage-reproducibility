# Reproducibility Notes

## Environment

The package was designed for Python 3.10 or later. The minimal dependencies are listed in `requirements.txt`.

## Reproduce all outputs

```bash
pip install -r requirements.txt
python src/bio_bandage_reproducible_study.py --all
```

Alternative:

```bash
python run_all.py
```

## Expected outputs

The command regenerates:

- CSV files in `data/`
- PNG figures in `figures/`
- `reproducibility_summary.json`

## Reproducibility limits

The code is a lightweight reconstructed surrogate aligned with the manuscript-level outputs. It is not the unavailable full FEM/PDE implementation. It is suitable for transparent result reconstruction, reviewer checking, and follow-up code development.

## Suggested verification

After running the code, check that:

1. `data/doe_runs.csv` contains 1296 rows.
2. `figures/fig1_permeation_temperature_response.png` exists.
3. `figures/fig2_pain_trajectory_across_arms.png` exists.
4. `figures/fig3_optimization_tradeoff.png` exists.
5. `figures/fig5_ablation_composite_score.png` exists.
6. `figures/fig7_robustness_curves.png` exists.
7. `figures/fig8_tornado_sensitivity.png` exists.
