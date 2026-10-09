# bio-bandage-reproducibility — release v2.0.0

Finite-element solver, simulation data and analysis scripts for the manuscript
*"Coupled Cooling, Diclofenac Transport and Compression–Contact Modelling for the In Silico Design of a
Multilayer Cryo-Compression Bandage"* (revision R1, SAJCE-D-26-00510).

## What this release contains

| Folder | Content |
|---|---|
| `src/bandage_fem.py` | 1D linear finite-element solver: Pennes bioheat, enthalpy-form apparent-heat-capacity phase change (raised-cosine kernel), pressure-dependent contact resistance, layer-specific Arrhenius diclofenac diffusion, partitioning, contact-limited transfer. Backward Euler + Newton, tridiagonal solves (Numba). |
| `src/study.py` | Penalties, objective `J`, burden `B`, `R_B`, `S_T`, `S_C`, `ΔT_peak`, `S_rank`, scenarios A1–A4. |
| `src/verify.py` | Analytical steady-state check, quasi-steady flux check, mass balance, mesh and time-step refinement. |
| `src/doe.py` | Full-factorial DOE: **all 3 × 1296 = 3888 runs** (Ice, PCM-A, PCM-B). |
| `src/optimize_designs.py` | Multistart bounded Powell optimisation with the hard skin constraint and margin ε. |
| `src/analysis.py`, `weights.py` | ε-margin sizing, lexicographic selection, scenarios, ablation, robustness, sensitivity, Pareto extraction, weight checks. |
| `src/figures.py`, `fig_permeation.py`, `graphical_abstract.py`, `make_tables.py` | Every figure and every LaTeX table of the manuscript, written directly from the result files. |
| `results/` | All outputs: `doe_runs.csv` (3888 runs), `doe_feasible_pareto.csv` (Pareto flags), `optimization_runs*.csv` (every start, start point, evaluations), verification, scenario, ablation, robustness, sensitivity and weight files. |

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py
```

## Scope

Release v2.0.0 supersedes v1.0.0, which contained the plotting scripts only. Every number in revision R1
is produced by the code in `src/`. All results are in silico model predictions; no clinical,
animal, ex vivo or bench-top data are involved.
