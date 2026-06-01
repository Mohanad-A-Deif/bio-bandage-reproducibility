# Dataset Card

## Dataset name

Bio-Bandage Reproducibility Synthetic Dataset

## Dataset type

Reconstructed synthetic computational dataset.

## Intended use

The dataset is intended to reproduce manuscript-level figures, tables, design rankings, DOE summaries, sensitivity analyses, and robustness analyses for an in-silico diclofenac cryocompression bandage framework.

## Not intended use

The dataset must not be used as evidence of clinical efficacy, patient outcomes, animal testing, ex-vivo permeation, or bench-top validation.

## Source and construction

The associated manuscript describes a fully in-silico computational framework. The plotting notebook used synthetic or hard-coded arrays. This repository reconstructs a transparent synthetic dataset that is numerically aligned with the manuscript and plotting outputs.

## Main files

- `doe_runs.csv`: 1296-run synthetic design-of-experiments grid.
- `optimized_designs.csv`: optimization-derived design candidates.
- `pareto_front_designs.csv`: representative Pareto/frontier designs.
- `arm_outcomes.csv`: scenario-level outcome summary.
- `arm_trajectories.csv`: normalized 14-day burden trajectories.
- `robustness_results.csv`: robustness analysis table.
- `robustness_curves.csv`: robustness curves used for plotting.
- `sensitivity_summary.csv`: parameter sensitivity summary.
- `sensitivity_tornado.csv`: tornado-plot input values.
- `baseline_forward_timeseries.csv`: synthetic 24-hour forward-model trajectory.

## Ethical and privacy considerations

No human participants are represented. No patient-level or personal data are included.

## Recommended wording

Use:

> We used a reconstructed synthetic computational dataset to reproduce the reported in-silico figures and tables.

Avoid:

> We used clinical data.

> We validated the treatment experimentally.

> The dataset proves therapeutic efficacy.
