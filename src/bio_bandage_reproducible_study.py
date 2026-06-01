#!/usr/bin/env python3
"""
Reproducible synthetic computational package for the multilayer diclofenac
cryo-compression bandage study.

Important scientific note
-------------------------
The manuscript and the attached plotting notebook describe an entirely in-silico
study. No raw clinical, animal, ex-vivo, or bench-top dataset was attached. The
notebook used to draw the result figures contains synthetic/hard-coded arrays.

This script reconstructs a transparent synthetic dataset that is numerically
consistent with the manuscript tables and the plotting notebook, then regenerates
all result CSV files and manuscript-style figures.

It is intended for reproducibility, reviewer checking, and manuscript-result
reconstruction. It should not be presented as measured clinical or experimental
raw data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# =============================================================================
# Paths and global constants
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

RNG_SEED_PERMEATION = 7
RNG_SEED_CANDIDATES = 12

# Objective weights reported in the manuscript.
OBJECTIVE_WEIGHTS = {
    "w_T": 0.40,
    "w_C": 0.32,
    "w_p": 0.10,
    "w_u": 0.13,
    "w_L": 0.05,
}

# Conservative post-processing ranking weights used in this reconstruction.
# These are chosen to reproduce the reported qualitative rankings and table values.
RANKING_WEIGHTS = {
    "lambda_R": 0.30,
    "lambda_ROM": 0.20,
    "lambda_strength": 0.17,
    "lambda_skin": 0.18,
    "lambda_dose": 0.08,
    "lambda_temp": 0.07,
}

P_MAX_MMHG = 50.0
L_REF_MM = 4.0


# =============================================================================
# Plotting style
# =============================================================================

def set_academic_style() -> None:
    """Nature/IEEE-inspired style matching the uploaded plotting notebook."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def savefig(filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / filename, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# Core lightweight digital-twin functions
# =============================================================================

def clipped_square_lower(value: np.ndarray | float, lower: float) -> np.ndarray | float:
    return np.maximum(lower - value, 0.0) ** 2


def clipped_square_upper(value: np.ndarray | float, upper: float) -> np.ndarray | float:
    return np.maximum(value - upper, 0.0) ** 2


def arrhenius_diffusivity(T_c: np.ndarray | float, D_ref: float, E_kj_mol: float, T_ref_c: float = 32.0) -> np.ndarray | float:
    """Arrhenius-form diffusivity used by the manuscript equation."""
    R = 8.314462618  # J mol^-1 K^-1
    T = np.asarray(T_c) + 273.15
    T_ref = T_ref_c + 273.15
    E = E_kj_mol * 1000.0
    return D_ref * np.exp(-(E / R) * (1.0 / T - 1.0 / T_ref))


def contact_resistance(p_mmhg: float, r0: float = 1.0, beta: float = 0.025) -> float:
    """Monotone contact-resistance model: R_c(p)=R_c0 exp(-beta p)."""
    return float(r0 * np.exp(-beta * p_mmhg))


def contact_efficiency(p_mmhg: float, eta_min: float = 0.55, eta_max: float = 1.0, beta: float = 0.055) -> float:
    """Effective interface mass-transfer/contact efficiency."""
    return float(eta_min + (eta_max - eta_min) * (1.0 - np.exp(-beta * p_mmhg)))


def permeation_model(t_h: np.ndarray | float, pmax: float, k_rate: float) -> np.ndarray | float:
    """Saturation-like cumulative diclofenac permeation model."""
    return pmax * (1.0 - np.exp(-k_rate * np.asarray(t_h)))


@dataclass(frozen=True)
class Design:
    l_cool_mm: float
    l_res_mm: float
    l_mem_mm: float
    l_gel_mm: float
    pressure_mmhg: float
    tau_on_min: float
    tau_off_min: float
    wear_h: float = 8.0
    mem_scale: float = 1.0
    medium: str = "PCM-A"

    @property
    def duty_fraction(self) -> float:
        if self.tau_off_min <= 0:
            return 1.0
        return self.tau_on_min / (self.tau_on_min + self.tau_off_min)


def surrogate_forward(design: Design) -> Dict[str, float]:
    """
    Compact surrogate of the coupled thermo--pharmaco--mechanical solver.

    The manuscript describes FEM/PDE-constrained optimization. The attached
    notebook, however, only contains plotting arrays. This function is therefore
    a calibrated lightweight digital-twin surrogate: it preserves the reported
    directionality of thermal buffering, permeability, contact, dose, response,
    and control penalties without pretending to be the unavailable full FEM code.
    """
    contact = contact_efficiency(design.pressure_mmhg)
    duty = design.duty_fraction

    # Synthetic dose balance: reservoir, membrane scale, contact and wear increase dose;
    # thicker membrane decreases it. The constant calibrates nominal D2 around 1.11 mg/cm^2.
    dose = (
        0.70
        * (design.l_res_mm / 2.1) ** 0.42
        * (0.10 / max(design.l_mem_mm, 0.04)) ** 0.20
        * design.mem_scale ** 0.55
        * (design.wear_h / 7.5) ** 0.38
        * (0.80 + 0.40 * contact)
    )

    # Thermal deviation: PCM thickness and gel thickness buffer deviation;
    # aggressive cooling and high contact increase skin-window sensitivity.
    cooling_drive = duty * (design.tau_on_min / 15.0) ** 0.22
    pcm_buffer = (4.2 / design.l_cool_mm) ** 0.35
    gel_buffer = (1.0 / design.l_gel_mm) ** 0.12
    peak_temp_dev = 0.38 + 0.30 * cooling_drive * pcm_buffer * contact + 0.07 * abs(design.pressure_mmhg - 25) / 15
    peak_temp_dev *= gel_buffer

    # Response saturates with dose and contact but is penalized by thermal deviation.
    response = 58.0 + 19.0 * (1.0 - np.exp(-1.35 * dose)) + 4.0 * (contact - 0.78) - 2.0 * max(peak_temp_dev - 0.8, 0.0)
    response = float(np.clip(response, 52.0, 78.0))

    rom = float(np.clip(62 + 0.33 * response + 7.0 * np.exp(-peak_temp_dev), 60, 90))
    strength = float(np.clip(58 + 0.30 * response + 5.0 * contact, 55, 88))
    skin_safety = float(np.clip(9.3 - 0.55 * peak_temp_dev - 0.04 * max(design.pressure_mmhg - 30, 0), 7.0, 9.5))

    # Normalized penalties used by the objective.
    phi_T = min((peak_temp_dev / 1.5) ** 2, 1.5)
    dose_target = 1.10
    phi_C = min(((dose - dose_target) / 0.45) ** 2, 1.5)
    pressure_pen = (design.pressure_mmhg / P_MAX_MMHG) ** 2
    cooling_pen = duty ** 2
    layer_pen = (design.l_cool_mm / L_REF_MM) ** 2 + (design.l_res_mm / L_REF_MM) ** 2 + (design.l_gel_mm / L_REF_MM) ** 2
    objective = (
        OBJECTIVE_WEIGHTS["w_T"] * phi_T
        + OBJECTIVE_WEIGHTS["w_C"] * phi_C
        + OBJECTIVE_WEIGHTS["w_p"] * pressure_pen
        + OBJECTIVE_WEIGHTS["w_u"] * cooling_pen
        + OBJECTIVE_WEIGHTS["w_L"] * layer_pen
    )

    # Post-processing ranking score, larger is better.
    dose_burden = abs(dose - 1.10) / 0.55
    temp_burden = peak_temp_dev / 1.5
    score = (
        RANKING_WEIGHTS["lambda_R"] * response / 100.0
        + RANKING_WEIGHTS["lambda_ROM"] * rom / 100.0
        + RANKING_WEIGHTS["lambda_strength"] * strength / 100.0
        + RANKING_WEIGHTS["lambda_skin"] * skin_safety / 10.0
        - RANKING_WEIGHTS["lambda_dose"] * dose_burden
        - RANKING_WEIGHTS["lambda_temp"] * temp_burden
    )
    # Rescale onto manuscript score range.
    score = 0.73 + 0.23 * (score - 0.50) / 0.25
    score = float(np.clip(score, 0.55, 0.94))

    return {
        "contact_efficiency": contact,
        "duty_fraction": duty,
        "dose_mg_cm2": float(dose),
        "peak_temp_dev_C": float(peak_temp_dev),
        "response_reduction_pct": response,
        "rom_surrogate_pct": rom,
        "strength_surrogate_pct": strength,
        "skin_safety_10": skin_safety,
        "phi_T_norm": float(phi_T),
        "phi_C_norm": float(phi_C),
        "objective_J": float(objective),
        "score": float(score),
        "feasible": bool(peak_temp_dev <= 1.5 and 0.85 <= dose <= 1.45 and design.pressure_mmhg <= 50),
    }


def synthetic_forward_timeseries() -> pd.DataFrame:
    """Create a 24-h synthetic C1 forward-model time series."""
    t = np.linspace(0, 24, 24 * 12 + 1)  # 5-min reporting interval
    # Duty-cycle cooling episodes superimposed onto recovery dynamics.
    cycle = ((t * 60) % 64) < 32
    cooling = cycle.astype(float)
    skin_temp = 31.5 - 20.0 * cooling * (1 - np.exp(-t / 0.35)) * np.exp(-t / 10.0) + 0.6 * np.sin(2 * np.pi * t / 24)
    skin_temp = np.clip(skin_temp, 10.2, 33.0)
    muscle_temp = 36.0 - 4.2 * cooling * (1 - np.exp(-t / 1.4)) * np.exp(-t / 13.0) + 0.25 * np.sin(2 * np.pi * t / 24 + 0.4)
    D_eff = arrhenius_diffusivity(skin_temp, 4.7e-10, 22.4)
    flux = 0.040 * (D_eff / np.nanmax(D_eff)) * (0.6 + 0.4 * contact_efficiency(25))
    dt = np.diff(t, prepend=t[0])
    cumulative = np.cumsum(flux * dt)
    phi_T = clipped_square_lower(skin_temp, 10) + clipped_square_upper(skin_temp, 15) * 0.03 + clipped_square_upper(muscle_temp, 35) * 0.02
    phi_C = clipped_square_lower(cumulative / np.max(cumulative + 1e-12), 0.25) * 0.2
    return pd.DataFrame({
        "time_h": t,
        "cooling_on": cooling,
        "skin_temperature_C": skin_temp,
        "muscle_temperature_C": muscle_temp,
        "effective_diffusivity_m2_s": D_eff,
        "absorption_flux_mg_cm2_h": flux,
        "cumulative_mass_mg_cm2": cumulative,
        "temperature_penalty_raw": phi_T,
        "concentration_penalty_raw": phi_C,
    })


# =============================================================================
# Dataset generation
# =============================================================================

def write_static_tables() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        ["C1", "Synthetic baseline simulations", "temperature fields; concentration fields; absorption-side flux; cumulative delivered mass"],
        ["C2", "DOE and mechanism scenarios", "objective values; feasibility indicators; safety-window violations; comparative design rankings"],
        ["C3", "Robustness and sensitivity simulations", "parameter-sensitivity maps; robustness envelopes; uncertainty effects on design performance"],
        ["C4", "PDE-constrained optimization", "optimized layer thicknesses; cooling schedules; compression schedules; final objective values"],
    ], columns=["dataset", "computational_setting", "main_outputs"]).to_csv(DATA_DIR / "dataset_manifest.csv", index=False)

    pd.DataFrame([
        ["Cooling source (PCM/ice)", "L_cool", 4.0, 2.0, 8.0, "mm"],
        ["Thermoplastic/seal", "L_shell", 1.0, 0.5, 2.0, "mm"],
        ["Thermal interface hydrogel", "L_gel", 1.0, 0.5, 3.0, "mm"],
        ["Diclofenac reservoir hydrogel", "L_res", 1.0, 0.5, 3.0, "mm"],
        ["Rate-controlling membrane", "L_mem", 0.10, 0.05, 0.30, "mm"],
        ["Skin-contact wicking liner", "L_liner", 1.0, 0.5, 2.0, "mm"],
    ], columns=["layer", "symbol", "baseline", "lower_bound", "upper_bound", "unit"]).to_csv(DATA_DIR / "layer_bounds.csv", index=False)

    pd.DataFrame([
        ["PCM/ice pack", "0.5--2.2", "900--1000", "2000--4200"],
        ["Thermoplastic shell", "0.20--0.25", "900--1200", "1500--2000"],
        ["Water-gel interface", "0.50--0.65", "950--1050", "3800--4200"],
        ["Drug hydrogel", "0.45--0.60", "950--1050", "3500--4200"],
        ["Membrane", "0.15--0.25", "1100--1300", "1200--1800"],
        ["Wicking liner", "0.03--0.06", "200--400", "1200--1800"],
        ["Skin effective", "0.30--0.40", "1050--1200", "3000--3600"],
        ["Fat effective", "0.18--0.25", "850--950", "2000--2500"],
        ["Muscle effective", "0.45--0.60", "1050--1100", "3400--3800"],
    ], columns=["layer", "k_W_mK", "rho_kg_m3", "c_J_kgK"]).to_csv(DATA_DIR / "thermal_parameters.csv", index=False)

    pd.DataFrame([
        ["Ice reference", "0", "334", "0.5--1.5"],
        ["PCM-A mild cooling", "8--12", "150--220", "1--3"],
        ["PCM-B stronger cooling", "4--8", "150--220", "1--3"],
    ], columns=["cooling_medium", "Tm_C", "Lf_kJ_kg", "deltaT_C"]).to_csv(DATA_DIR / "pcm_parameters.csv", index=False)

    pd.DataFrame([
        ["Initial reservoir concentration", "C0", "5--20", "mg/mL"],
        ["Reservoir diffusivity at Tref", "D_res_ref", "1e-10--1e-9", "m2/s"],
        ["Membrane diffusivity at Tref", "D_mem_ref", "1e-12--1e-10", "m2/s"],
        ["Skin-barrier diffusivity at Tref", "D_skin_ref", "1e-13--1e-11", "m2/s"],
        ["Layer-specific activation energy", "E_D", "10--40", "kJ/mol"],
        ["Partition coefficient reservoir to membrane", "K_res_to_mem", "0.2--2", "dimensionless"],
        ["Partition coefficient membrane to skin", "K_mem_to_skin", "0.05--1", "dimensionless"],
        ["Effective clearance coefficient", "k_clr", "1e-7--1e-5", "m/s"],
    ], columns=["parameter", "symbol", "planning_range", "unit"]).to_csv(DATA_DIR / "drug_transport_parameters.csv", index=False)

    pd.DataFrame([
        ["w_T", 0.40, "Phi_T_hat", "temperature-window penalty"],
        ["w_C", 0.32, "Phi_C_hat", "drug-concentration-window penalty"],
        ["w_p", 0.10, "(p/pmax)^2", "compression-control effort"],
        ["w_u", 0.13, "(u/uref)^2", "cooling-control effort"],
        ["w_L", 0.05, "sum(L_j/Lref)^2", "layer-thickness regularization"],
    ], columns=["weight", "value", "associated_term", "interpretation"]).to_csv(DATA_DIR / "objective_weights.csv", index=False)

    pd.DataFrame([
        ["PCM melting temperature Tm", 8.6, "C"],
        ["PCM transition half-width deltaT", 1.7, "C"],
        ["PCM latent heat Lf", 182, "kJ/kg"],
        ["Reservoir diffusivity D_res_ref", 4.7e-10, "m2/s"],
        ["Lumped activation energy E_D", 22.4, "kJ/mol"],
        ["Partition coefficient K_res_to_mem", 0.71, "dimensionless"],
        ["Partition coefficient K_mem_to_skin", 0.19, "dimensionless"],
        ["Effective clearance coefficient k_clr", 3.1e-6, "m/s"],
        ["Thermal self-consistency RMSE", 0.28, "C"],
        ["Thermal self-consistency R2", 0.989, "dimensionless"],
        ["Permeation self-consistency RMSE", 0.023, "mg/cm2"],
        ["Permeation self-consistency R2", 0.997, "dimensionless"],
    ], columns=["quantity", "value", "unit"]).to_csv(DATA_DIR / "benchmark_parameters.csv", index=False)

    pd.DataFrame([
        ["Spatial discretization", "1D conforming FEM, quadratic elements p=2"],
        ["Mesh resolution", ">=40 elements per layer; >=120 within membrane-skin stack"],
        ["Time horizon", "Tf=24 h per simulated day; acute window aggregated over 14 days"],
        ["Time step", "dt=1 s thermal; dt=5 s drug; substepping allowed"],
        ["Nonlinear solver", "Newton method with line search; maximum 25 iterations"],
        ["Linear solver", "GMRES with ILU preconditioner"],
        ["Tolerances", "relative 1e-8; absolute 1e-10"],
        ["Optimization method", "projected L-BFGS or SQP"],
        ["Maximum optimization iterations", "300"],
        ["Stopping criteria", "||grad J||<=1e-5 or DeltaJ/J<=1e-6"],
    ], columns=["item", "value"]).to_csv(DATA_DIR / "solver_settings.csv", index=False)


def generate_permeation_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(RNG_SEED_PERMEATION)
    times = np.array([0, 0.5, 1, 2, 3, 4, 6, 8], dtype=float)
    temperatures = [15, 22, 32]
    P_inf = {15: 0.78, 22: 1.02, 32: 1.38}
    k_rate = {15: 0.23, 22: 0.31, 32: 0.44}
    t_dense = np.linspace(0, 8, 400)

    point_rows = []
    curve_rows = []
    for T in temperatures:
        y_model = permeation_model(times, P_inf[T], k_rate[T])
        y_ref = np.clip(y_model + np.random.normal(0, 0.025, size=len(times)), 0, None)
        for t, pred, ref in zip(times, y_model, y_ref):
            point_rows.append([T, t, pred, ref, ref - pred])
        y_dense = permeation_model(t_dense, P_inf[T], k_rate[T])
        for t, val in zip(t_dense, y_dense):
            curve_rows.append([T, t, val])

    points = pd.DataFrame(point_rows, columns=["temperature_C", "time_h", "model_prediction_mg_cm2", "synthetic_reference_mg_cm2", "residual_mg_cm2"])
    curves = pd.DataFrame(curve_rows, columns=["temperature_C", "time_h", "model_prediction_mg_cm2"])
    points.to_csv(DATA_DIR / "permeation_temperature_points.csv", index=False)
    curves.to_csv(DATA_DIR / "permeation_temperature_curves.csv", index=False)
    return points, curves


def generate_result_tables() -> None:
    pd.DataFrame([
        ["A1", "Full coupled model", 5.8, 71, 88, 84, 8.9],
        ["A2", "Cooling + compression", 7.1, 58, 79, 74, 8.7],
        ["A3", "Drug + compression", 7.8, 54, 75, 72, 8.5],
        ["A4", "Baseline contact model", 10.2, 31, 61, 57, 8.3],
    ], columns=["scenario_id", "scenario", "time_to_response_days", "response_reduction_day14_pct", "rom_day14_pct", "strength_day14_pct", "skin_safety_10"]).to_csv(DATA_DIR / "arm_outcomes.csv", index=False)

    days = np.array([0, 3, 7, 10, 14])
    trajectories = {
        "A1_Full_coupled_model": [1.00, 0.78, 0.52, 0.37, 0.29],
        "A2_Cooling_compression": [1.00, 0.83, 0.62, 0.50, 0.42],
        "A3_Drug_compression": [1.00, 0.86, 0.66, 0.55, 0.46],
        "A4_Baseline_contact_model": [1.00, 0.93, 0.82, 0.75, 0.69],
    }
    rows = []
    for name, ys in trajectories.items():
        for d, y in zip(days, ys):
            rows.append([name, d, y])
    pd.DataFrame(rows, columns=["scenario", "day", "normalized_surrogate_burden"]).to_csv(DATA_DIR / "arm_trajectories.csv", index=False)

    pd.DataFrame([
        ["D1", 2.4, 3.8, 1.00, 28, 8.0, 74.2, 0.82, 1.18, 0.912],
        ["D2", 2.1, 4.2, 0.93, 32, 7.5, 72.9, 0.61, 1.11, 0.905],
        ["D3", 2.8, 3.5, 1.08, 24, 8.5, 75.1, 1.04, 1.26, 0.896],
        ["D4", 1.9, 4.5, 0.88, 35, 7.0, 70.8, 0.48, 1.03, 0.894],
        ["D5", 2.5, 3.2, 1.15, 22, 9.0, 76.0, 1.28, 1.34, 0.881],
    ], columns=["design", "hydrogel_mm", "pcm_mm", "membrane_scale", "cooling_min", "wear_h", "response_reduction_day14_pct", "peak_temp_dev_C", "dose_mg_cm2", "score"]).to_csv(DATA_DIR / "optimized_designs.csv", index=False)

    pd.DataFrame([
        ["Full coupled design", 72.9, 0.61, 1.11, 86, 0.905],
        ["No thermal coupling", 61.8, 0.24, 0.89, 77, 0.792],
        ["No membrane regulation", 66.5, 0.96, 1.42, 80, 0.781],
        ["No compression/contact coupling", 64.1, 0.63, 1.09, 74, 0.768],
        ["Drug-only transport baseline", 57.4, 0.18, 0.83, 71, 0.702],
    ], columns=["configuration", "response_reduction_day14_pct", "peak_temp_dev_C", "dose_mg_cm2", "rom_surrogate_pct", "score"]).to_csv(DATA_DIR / "ablation_results.csv", index=False)

    pd.DataFrame([
        ["P1", 0.92, 66.8, 0.34, 78, "Conservative low-dose design"],
        ["P2", 1.03, 69.9, 0.47, 81, "Balanced low-risk design"],
        ["P3", 1.11, 72.9, 0.61, 86, "Selected compromise design"],
        ["P4", 1.23, 74.6, 0.83, 87, "Response-oriented design"],
        ["P5", 1.36, 76.1, 1.09, 88, "Higher-response but less conservative design"],
    ], columns=["design", "dose_mg_cm2", "response_reduction_day14_pct", "peak_temp_dev_C", "rom_surrogate_pct", "interpretation"]).to_csv(DATA_DIR / "pareto_front_designs.csv", index=False)

    pd.DataFrame([
        ["Nominal design", 72.9, 0.61, 1.11, 0.905, "Reference configuration"],
        ["Contact efficiency -10%", 70.8, 0.64, 1.08, 0.878, "Mild degradation"],
        ["Contact efficiency -20%", 68.1, 0.68, 1.03, 0.842, "Larger contact-related loss"],
        ["Permeability +10%", 73.6, 0.71, 1.22, 0.884, "Higher response with higher burden"],
        ["Permeability -10%", 70.2, 0.56, 1.00, 0.867, "Lower transport and response"],
        ["Combined perturbation", 69.1, 0.73, 1.17, 0.851, "Moderate combined degradation"],
    ], columns=["scenario", "response_reduction_day14_pct", "peak_temp_dev_C", "dose_mg_cm2", "score", "interpretation"]).to_csv(DATA_DIR / "robustness_results.csv", index=False)

    pd.DataFrame([
        ["contact_efficiency", 0, 0.905],
        ["contact_efficiency", -10, 0.878],
        ["contact_efficiency", -20, 0.842],
        ["membrane_permeability", -10, 0.867],
        ["membrane_permeability", 0, 0.905],
        ["membrane_permeability", 10, 0.884],
    ], columns=["perturbation_type", "perturbation_pct", "score"]).to_csv(DATA_DIR / "robustness_curves.csv", index=False)

    sensitivity = pd.DataFrame([
        ["Membrane permeability scaling", -0.041, 0.029, 0.070, 1, "Highest local influence on delivery-safety balance"],
        ["Initial cooling duration", -0.031, 0.034, 0.065, 2, "Strong operational influence on thermal response"],
        ["Reservoir diffusivity D_res_ref", -0.023, 0.026, 0.049, 3, "Moderate-high transport sensitivity"],
        ["PCM thickness", -0.027, 0.021, 0.048, 4, "Strong thermal-design influence"],
        ["Hydrogel thickness", -0.026, 0.014, 0.040, 5, "Moderate design sensitivity"],
        ["Effective clearance k_clr", -0.017, 0.019, 0.036, 6, "Moderate exposure-profile sensitivity"],
        ["Daily wear time", -0.020, 0.011, 0.031, 7, "Moderate operational sensitivity"],
        ["Activation energy E_D", -0.012, 0.016, 0.028, 8, "Lower but non-negligible thermal-transport sensitivity"],
    ], columns=["parameter", "minus_10pct_delta_score", "plus_10pct_delta_score", "sensitivity_range", "rank", "interpretation"])
    sensitivity.to_csv(DATA_DIR / "sensitivity_tornado.csv", index=False)
    sensitivity[["parameter", "sensitivity_range", "rank", "interpretation"]].to_csv(DATA_DIR / "sensitivity_summary.csv", index=False)


def generate_doe_dataset() -> pd.DataFrame:
    levels = {
        "l_cool_mm": [2.0, 4.0, 8.0],
        "l_res_mm": [0.5, 1.0, 2.0],
        "l_mem_mm": [0.05, 0.10, 0.30],
        "l_gel_mm": [0.5, 1.0, 2.0],
        "pressure_mmhg": [10, 20, 30, 40],
        "duty_cycle_label": ["10/10", "15/15", "20/20", "continuous"],
    }
    duty_map = {
        "10/10": (10, 10),
        "15/15": (15, 15),
        "20/20": (20, 20),
        "continuous": (20, 0),
    }
    rows = []
    run_id = 1
    for combo in itertools.product(*levels.values()):
        l_cool, l_res, l_mem, l_gel, p, duty_label = combo
        tau_on, tau_off = duty_map[duty_label]
        design = Design(l_cool, l_res, l_mem, l_gel, p, tau_on, tau_off, wear_h=8.0, mem_scale=1.0, medium="PCM-A")
        out = surrogate_forward(design)
        rows.append({
            "run_id": run_id,
            "cooling_medium": design.medium,
            "L_cool_mm": l_cool,
            "L_res_mm": l_res,
            "L_mem_mm": l_mem,
            "L_gel_mm": l_gel,
            "pressure_mmhg": p,
            "duty_cycle": duty_label,
            "tau_on_min": tau_on,
            "tau_off_min": tau_off,
            **out,
        })
        run_id += 1
    doe = pd.DataFrame(rows)
    doe.to_csv(DATA_DIR / "doe_runs.csv", index=False)
    return doe


def generate_candidate_pool() -> pd.DataFrame:
    np.random.seed(RNG_SEED_CANDIDATES)
    n = 40
    temp_dev = np.random.uniform(0.4, 1.5, n)
    response = 68 + 8 * (1 - (temp_dev - 0.4) / (1.5 - 0.4)) + np.random.normal(0, 1.8, n)
    df = pd.DataFrame({
        "design": [f"C{i+1:02d}" for i in range(n)],
        "group": "candidate_pool",
        "peak_temp_dev_C": temp_dev,
        "response_reduction_day14_pct": response,
    })
    top = pd.DataFrame({
        "design": ["D1", "D2", "D3", "D4", "D5"],
        "group": "top_optimized_design",
        "peak_temp_dev_C": [0.82, 0.61, 1.04, 0.48, 1.28],
        "response_reduction_day14_pct": [74.2, 72.9, 75.1, 70.8, 76.0],
    })
    full = pd.concat([df, top], ignore_index=True)
    full.to_csv(DATA_DIR / "optimization_candidate_pool.csv", index=False)
    return full


# =============================================================================
# Figure generation
# =============================================================================

def plot_permeation(points: pd.DataFrame, curves: pd.DataFrame) -> None:
    set_academic_style()
    line_colors = {15: "#1F3A5F", 22: "#5C3B2E", 32: "#2F5D50"}
    marker_colors = {15: "#2C4F80", 22: "#7A4B3A", 32: "#3E7565"}
    markers = {15: "o", 22: "s", 32: "^"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for T in [15, 22, 32]:
        c = curves[curves["temperature_C"] == T]
        p = points[points["temperature_C"] == T]
        ax.plot(c["time_h"], c["model_prediction_mg_cm2"], linewidth=2.6, color=line_colors[T], label=f"Model prediction, {T}°C")
        ax.scatter(p["time_h"], p["synthetic_reference_mg_cm2"], s=52, marker=markers[T], facecolor=marker_colors[T], edgecolor="black", linewidth=0.7, zorder=3, label=f"Synthetic reference points, {T}°C")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"Cumulative permeation (mg/cm$^2$)")
    ax.set_title("Temperature-dependent diclofenac permeation self-consistency", pad=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.22)
    leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95, edgecolor="black", ncol=1)
    leg.get_frame().set_linewidth(0.9)
    plt.tight_layout()
    savefig("fig1_permeation_temperature_response.png")


def plot_arm_trajectories() -> None:
    set_academic_style()
    df = pd.read_csv(DATA_DIR / "arm_trajectories.csv")
    colors = {"A1": "#1F3A5F", "A2": "#5C3B2E", "A3": "#2F5D50", "A4": "#6A3D5A"}
    markers = {"A1": "o", "A2": "s", "A3": "^", "A4": "D"}
    labels = {
        "A1_Full_coupled_model": "A1 (Full coupled model)",
        "A2_Cooling_compression": "A2 (Cooling + compression)",
        "A3_Drug_compression": "A3 (Drug + compression)",
        "A4_Baseline_contact_model": "A4 (Baseline contact model)",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for scenario, label in labels.items():
        sub = df[df["scenario"] == scenario]
        key = scenario[:2]
        ax.plot(sub["day"], sub["normalized_surrogate_burden"], marker=markers[key], markersize=6.5, linewidth=2.6, color=colors[key], markerfacecolor=colors[key], markeredgecolor="black", markeredgewidth=0.7, label=label)
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalized surrogate burden")
    ax.set_title("Model-derived surrogate-burden trajectories", pad=10)
    ax.set_xticks([0, 3, 7, 10, 14])
    ax.set_ylim(0.2, 1.05)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.22)
    leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95, edgecolor="black", ncol=1)
    leg.get_frame().set_linewidth(0.9)
    plt.tight_layout()
    savefig("fig2_pain_trajectory_across_arms.png")


def plot_optimization_tradeoff() -> None:
    set_academic_style()
    df = pd.read_csv(DATA_DIR / "optimization_candidate_pool.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cand = df[df["group"] == "candidate_pool"]
    top = df[df["group"] == "top_optimized_design"]
    ax.scatter(cand["peak_temp_dev_C"], cand["response_reduction_day14_pct"], s=48, alpha=0.72, color="#5A6C7D", edgecolor="black", linewidth=0.45, label="Candidate designs")
    ax.scatter(top["peak_temp_dev_C"], top["response_reduction_day14_pct"], s=96, marker="D", color="#7A3E2B", edgecolor="black", linewidth=0.75, label="Top optimized designs")
    for _, row in top.iterrows():
        ax.text(row["peak_temp_dev_C"] + 0.025, row["response_reduction_day14_pct"] + 0.12, row["design"], fontsize=11, color="#1F1F1F")
    ax.set_xlabel(r"Peak skin-temperature deviation ($^\circ$C)")
    ax.set_ylabel("Response-surrogate reduction at day 14 (%)")
    ax.set_title("Optimization trade-off between response and thermal deviation", pad=10)
    ax.set_xlim(0.3, 1.6)
    ax.set_ylim(66, 79)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.22)
    leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95, edgecolor="black")
    leg.get_frame().set_linewidth(0.9)
    plt.tight_layout()
    savefig("fig3_optimization_tradeoff.png")


def plot_bandage_configuration() -> None:
    set_academic_style()
    layers = [
        ("PCM cooling\n4.2 mm", 4.2, "#1F3A5F"),
        ("Shell\n1.0 mm", 1.0, "#4F5866"),
        ("Hydrogel interface\n1.0 mm", 1.0, "#2F5D50"),
        ("Drug reservoir\n2.1 mm", 2.1, "#5C3B2E"),
        ("Membrane\n0.10 mm", 0.25, "#7A3E2B"),
        ("Wicking liner\n1.0 mm", 1.0, "#6A3D5A"),
        ("Skin/fat/muscle\ncomputational tissue", 3.0, "#B08A70"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = 0.0
    for label, width, color in layers:
        ax.add_patch(plt.Rectangle((x, 0.25), width, 1.4, facecolor=color, edgecolor="black", linewidth=0.9, alpha=0.92))
        ax.text(x + width / 2, 0.95, label, ha="center", va="center", fontsize=10, color="white" if color != "#B08A70" else "black")
        x += width
    ax.annotate("Outer surface", xy=(0, 1.9), xytext=(0, 2.35), arrowprops=dict(arrowstyle="->", lw=1.0), ha="center")
    ax.annotate("Toward tissue depth", xy=(x, 1.9), xytext=(x - 1.0, 2.35), arrowprops=dict(arrowstyle="->", lw=1.0), ha="center")
    ax.set_xlim(-0.4, x + 0.4)
    ax.set_ylim(0, 2.65)
    ax.set_xlabel("Through-thickness computational coordinate")
    ax.set_yticks([])
    ax.set_title("Optimization-selected multilayer cryo-bandage configuration", pad=10)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    savefig("cryo-bandage configuration.png")


def plot_mechanism_schematic() -> None:
    set_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    titles = ["(a) Hamstring strain", "(b) Multilayer bandage", "(c) Coupled solver"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    # Panel a
    ax = axes[0]
    ax.plot([0.15, 0.85], [0.5, 0.5], lw=7, color="#5C3B2E", solid_capstyle="round")
    ax.plot([0.45, 0.55], [0.46, 0.54], lw=2.5, color="black")
    ax.text(0.5, 0.25, "localized strain\nregion", ha="center", va="center", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Panel b
    ax = axes[1]
    y0 = 0.12
    labels = ["PCM", "shell", "gel", "drug", "membrane", "liner", "skin"]
    colors = ["#1F3A5F", "#4F5866", "#2F5D50", "#5C3B2E", "#7A3E2B", "#6A3D5A", "#B08A70"]
    for i, (lab, col) in enumerate(zip(labels, colors)):
        ax.add_patch(plt.Rectangle((0.18, y0 + i*0.10), 0.64, 0.08, facecolor=col, edgecolor="black", lw=0.7))
        ax.text(0.5, y0 + i*0.10 + 0.04, lab, ha="center", va="center", fontsize=9, color="white" if col != "#B08A70" else "black")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Panel c
    ax = axes[2]
    boxes = [(0.08, 0.68, "Heat\nPDE"), (0.58, 0.68, "Drug\nPDE"), (0.08, 0.25, "Contact /\npressure"), (0.58, 0.25, "Objective\nranking")]
    for x, y, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.34, 0.18, facecolor="#E8ECEF", edgecolor="black", lw=0.9))
        ax.text(x+0.17, y+0.09, txt, ha="center", va="center", fontsize=10)
    for start, end in [((0.42,0.77),(0.58,0.77)), ((0.25,0.68),(0.25,0.43)), ((0.75,0.68),(0.75,0.43)), ((0.42,0.34),(0.58,0.34))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    savefig("mechanism.png")


def plot_ablation() -> None:
    set_academic_style()
    df = pd.read_csv(DATA_DIR / "ablation_results.csv")
    configs = ["Full coupled", "No thermal\ncoupling", "No membrane\nregulation", "No mechanical\nsupport", "Drug-only\nbaseline"]
    scores = df["score"].values
    bar_colors = ["#1F3A5F", "#5C3B2E", "#2F5D50", "#6A3D5A", "#4F5866"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    bars = ax.bar(configs, scores, color=bar_colors, edgecolor="black", linewidth=0.9, width=0.68)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=12, color="black")
    ax.set_ylabel("Composite performance score")
    ax.set_title("Ablation of coupled thermo--pharmaco--mechanical mechanisms", pad=10)
    ax.set_ylim(0.65, 0.95)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=False, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.22)
    plt.tight_layout()
    savefig("fig5_ablation_composite_score.png")


def plot_robustness() -> None:
    set_academic_style()
    df = pd.read_csv(DATA_DIR / "robustness_curves.csv")
    contact = df[df["perturbation_type"] == "contact_efficiency"]
    perm = df[df["perturbation_type"] == "membrane_permeability"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(contact["perturbation_pct"], contact["score"], marker="o", markersize=6.5, linewidth=2.6, color="#1F3A5F", markerfacecolor="#1F3A5F", markeredgecolor="black", markeredgewidth=0.7, label="Contact efficiency perturbation")
    ax.plot(perm["perturbation_pct"], perm["score"], marker="s", markersize=6.2, linewidth=2.6, color="#5C3B2E", markerfacecolor="#5C3B2E", markeredgecolor="black", markeredgewidth=0.7, label="Membrane permeability perturbation")
    ax.set_xlabel("Perturbation level (%)")
    ax.set_ylabel("Composite performance score")
    ax.set_title("Robustness of the selected design under representative perturbations", pad=10)
    ax.set_ylim(0.83, 0.92)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.22)
    leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95, edgecolor="black")
    leg.get_frame().set_linewidth(0.9)
    plt.tight_layout()
    savefig("fig7_robustness_curves.png")


def plot_sensitivity() -> None:
    set_academic_style()
    df = pd.read_csv(DATA_DIR / "sensitivity_tornado.csv")
    params = [
        "Membrane permeability scaling",
        "Initial cooling duration",
        "PCM thickness",
        r"Reservoir diffusivity $D_0$",
        "Hydrogel thickness",
        r"Effective clearance $k_{clr}$",
        "Daily wear time",
        r"Activation energy $E_D$",
    ]
    # Match the exact order and values of the uploaded notebook.
    neg = np.array([-0.041, -0.031, -0.027, -0.023, -0.026, -0.017, -0.020, -0.012])
    pos = np.array([0.029, 0.034, 0.021, 0.026, 0.014, 0.019, 0.011, 0.016])
    y = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.barh(y, neg, color="#5C3B2E", edgecolor="black", linewidth=0.8, height=0.62, label="-10% perturbation")
    ax.barh(y, pos, color="#1F3A5F", edgecolor="black", linewidth=0.8, height=0.62, label="+10% perturbation")
    ax.axvline(0, color="#1F1F1F", linewidth=1.2)
    for yi, v in zip(y, neg):
        ax.text(v - 0.0015, yi, f"{v:.3f}", va="center", ha="right", fontsize=8, color="black")
    for yi, v in zip(y, pos):
        ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", ha="left", fontsize=8, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=12)
    ax.set_xlabel("Change in composite performance score")
    ax.set_title("Local sensitivity of the selected design (±10% parameter perturbation)", pad=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(True, axis="x", which="major", linestyle="--", linewidth=0.6, alpha=0.22)
    ax.invert_yaxis()
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.95, edgecolor="black")
    leg.get_frame().set_linewidth(0.9)
    plt.tight_layout()
    savefig("fig8_tornado_sensitivity.png")


def generate_all_figures(points: pd.DataFrame, curves: pd.DataFrame) -> None:
    plot_mechanism_schematic()
    plot_permeation(points, curves)
    plot_arm_trajectories()
    plot_optimization_tradeoff()
    plot_bandage_configuration()
    plot_ablation()
    plot_robustness()
    plot_sensitivity()


def write_readme() -> None:
    text = """# Bio Bandage reproducible synthetic dataset

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
"""
    (DATA_DIR / "README.md").write_text(text, encoding="utf-8")
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def run_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_static_tables()
    points, curves = generate_permeation_data()
    generate_result_tables()
    generate_doe_dataset()
    generate_candidate_pool()
    synthetic_forward_timeseries().to_csv(DATA_DIR / "baseline_forward_timeseries.csv", index=False)
    generate_all_figures(points, curves)
    write_readme()

    summary = {
        "csv_files": sorted([p.name for p in DATA_DIR.glob("*.csv")]),
        "figure_files": sorted([p.name for p in FIG_DIR.glob("*.png")]),
        "doe_runs": int(pd.read_csv(DATA_DIR / "doe_runs.csv").shape[0]),
        "interpretation": "reconstructed synthetic in-silico dataset; not measured raw data",
    }
    (ROOT / "reproducibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reconstructed synthetic dataset and figures for the Bio Bandage computational study.")
    parser.add_argument("--all", action="store_true", help="Generate all CSV datasets and figures.")
    args = parser.parse_args()
    if args.all:
        run_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
