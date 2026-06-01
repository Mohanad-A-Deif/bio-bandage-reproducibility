# Bio Bandage reproducible synthetic dataset

This folder contains the reconstructed computational dataset used to reproduce the manuscript tables and figures.

## Critical interpretation
The manuscript states that the work is entirely in silico. The uploaded notebook contained plotting code with synthetic/hard-coded arrays, not the full raw FEM/PDE solver output. Therefore, these CSV files are a transparent reconstructed synthetic dataset aligned with the manuscript and the plotting notebook. They are not clinical, animal, ex-vivo, or bench-top measurements.

## Main files
- `benchmark_parameters.csv`: nominal synthetic benchmark values and self-consistency metrics.
- `permeation_temperature_points.csv`: synthetic reference points used in Fig. 1.
- `permeation_temperature_curves.csv`: smooth model curves used in Fig. 1.
- `arm_outcomes.csv`: scenario-comparison table.
- `arm_trajectories.csv`: normalized burden trajectories for Fig. 2.
- `optimization_candidate_pool.csv`: candidate pool and top designs for Fig. 3.
- `optimized_designs.csv`: top optimization-derived designs.
- `ablation_results.csv`: mechanism-ablation table and Fig. 5 values.
- `pareto_front_designs.csv`: representative DOE/Pareto designs.
- `robustness_results.csv` and `robustness_curves.csv`: robustness table and Fig. 7 values.
- `sensitivity_summary.csv` and `sensitivity_tornado.csv`: sensitivity table and Fig. 8 values.
- `doe_runs.csv`: 1296-run synthetic DOE grid following the manuscript factor levels.
- `baseline_forward_timeseries.csv`: synthetic 24-h forward-model time series for C1.

## Recreate everything
From the package root, run:

```bash
python src/bio_bandage_reproducible_study.py --all
```

Figures will be written to `figures/` and CSV files to `data/`.
