"""Result figures for R1 (all generated from solver outputs in ../results)."""
import json
import numpy as np, pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import bandage_fem as bf
from bandage_fem import Params
from study import *
from figs_common import *

style()
R, F = "../results/", "../figures/"
sel = json.load(open(R + "selected_design.json"))
SEL = Params(**{k: v for k, v in sel.items() if k in Params.__dataclass_fields__})

# ------------------------------------------------ scenario time series (3 small multiples)
fig, axs = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True)
runs = {sc: bf.run(scenario(SEL, sc)) for sc in ["A1", "A2", "A3", "A4"]}
t = runs["A1"]["t"] / 3600; m = t <= 8
axs[0].plot(t[m], runs["A1"]["Ts"][m], "-", color=PAL[0], lw=1.6, label="With cooling (A1, A2)")
axs[0].plot(t[m], runs["A3"]["Ts"][m], "--", color=PAL[1], lw=1.6, label="Without cooling (A3, A4)")
axs[1].plot(t[m], runs["A1"]["Tmu"][m], "-", color=PAL[0], lw=1.6)
axs[1].plot(t[m], runs["A3"]["Tmu"][m], "--", color=PAL[1], lw=1.6)
axs[2].plot(t[m], runs["A1"]["Cabs"][m] * 1e3, "-", color=PAL[0], lw=1.6, label="A1 full coupled (cooling + drug)")
axs[2].plot(t[m], runs["A3"]["Cabs"][m] * 1e3, "--", color=PAL[1], lw=1.6, label="A3 drug + compression")
for k, (lo, hi) in enumerate([(T_MIN, T_MAX), (TMU_LOW, TMU_HIGH), (C_LOW * 1e3, C_HIGH * 1e3)]):
    axs[k].axhspan(lo, hi, color=GRID, alpha=0.7, lw=0, label="Target window" if k == 0 else None)
axs[0].set_ylabel("Skin-surface\ntemperature (°C)")
axs[1].set_ylabel("Muscle temperature,\n10 mm depth (°C)")
axs[2].set_ylabel("Absorption-side\nconcentration (µg mL$^{-1}$)")
axs[2].set_xlabel("Time from bandage application (h)")
axs[2].set_xlim(0, 8)
axs[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8.5)
axs[2].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5)
for a_, l in zip(axs, "abc"):
    a_.text(0.995, 0.97, f"({l})", transform=a_.transAxes, va="top", ha="right", fontsize=11, color=INK)
fig.align_ylabels(axs)
fig.tight_layout()
save(fig, F + "fig_scenarios_timeseries.png")

# ------------------------------------------------ DOE trade-off (corrects former Fig. 4 labelling)
doe = pd.read_csv(R + "doe_runs.csv")
par = pd.read_csv(R + "doe_feasible_pareto.csv")
cand = pd.read_csv(R + "candidate_designs.csv")
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for j, med in enumerate(["Ice", "PCM-A", "PCM-B"]):
    d = doe[(doe.medium == med) & (doe.Ts_min >= T_MIN)]
    jit = np.random.default_rng(j).uniform(-0.6, 0.6, len(d)) if med != "Ice" else 0
    ax.scatter(d.S_T + jit, d.R_B, s=10, color=PAL[j], alpha=0.35, lw=0, label=f"{med} (admissible)")
inf = doe[doe.Ts_min < T_MIN]
ax.scatter(inf.S_T, inf.R_B, s=12, marker="x", color="#8C8C8C", lw=0.7, alpha=0.6, label="Ice, violates $T_{skin}\\geq$10 °C")
p3 = par[par.pareto3]
ax.scatter(p3.S_T, p3.R_B, s=40, facecolor="none", edgecolor=PAL[3], lw=1.4, label="Non-dominated DOE runs")
c = cand[cand.label.str.match(r"^D\d$")]
ax.scatter(c.S_T, c.R_B, s=60, marker="D", color=PAL[4], edgecolor="white", lw=1.0, zorder=5)
r = c[c.selected.astype(bool)].iloc[0]
ax.scatter(r.S_T, r.R_B, s=200, marker="*", color=PAL[6], edgecolor="white", lw=1.0, zorder=6)
ax.annotate(f"D1–D5 (optimised); {r.label} selected", (r.S_T, r.R_B), xytext=(-150, 22), textcoords="offset points",
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("Skin thermal-window compliance score $S_T$ (%)")
ax.set_ylabel("Normalized burden-reduction index $R_B$ (%)")
ax.set_xlim(-3, 80); ax.set_ylim(0, 100)
h, l = ax.get_legend_handles_labels()
h += [Line2D([], [], marker="D", ls="", color=PAL[4], label="Optimised designs D1–D5"),
      Line2D([], [], marker="*", ls="", ms=12, color=PAL[6], label="Selected design")]
ax.legend(handles=h, loc="lower right", fontsize=8.2, ncol=1)
save(fig, F + "fig_doe_tradeoff.png")

# ------------------------------------------------ ablation (two panels: dose, S_rank)
ab = pd.read_csv(R + "ablation.csv")
lab = ["Full coupled", "No thermal\ncoupling", "No membrane\nregulation", "No compression/\ncontact coupling",
       "Drug-only\nbaseline"]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.8))
y = np.arange(len(ab))[::-1]
axs[0].barh(y, ab.M14, color=[PAL[6]] + [PAL[1]] * 4, height=0.6)
axs[0].axvline(M_TARGET, color=INK2, ls=":", lw=1.2)
axs[0].text(M_TARGET, -0.75, "target", ha="center", fontsize=8.5, color=INK2)
for yi, v in zip(y, ab.M14):
    axs[0].text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
axs[0].set_yticks(y); axs[0].set_yticklabels(lab, fontsize=9)
axs[0].set_xlabel("Predicted 14-day delivered dose (mg cm$^{-2}$)"); axs[0].set_xlim(0, 0.8)
axs[1].barh(y, ab.S_rank, color=[PAL[6]] + [PAL[1]] * 4, height=0.6)
for yi, v in zip(y, ab.S_rank):
    axs[1].text(max(v, 0) + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
axs[1].axvline(0, color=INK2, lw=0.8)
axs[1].set_yticks(y); axs[1].set_yticklabels([])
axs[1].set_xlabel("Composite ranking score $S_{rank}$"); axs[1].set_xlim(-0.1, 0.95)
for a, l in zip(axs, "ab"):
    a.text(-0.02, 1.04, f"({l})", transform=a.transAxes, fontsize=11)
    a.grid(axis="y", visible=False)
save(fig, F + "fig_ablation.png")

# ------------------------------------------------ robustness curves
rc = pd.read_csv(R + "robustness_curves.csv")
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
for j, (kind, lbl) in enumerate([("contact", "Contact-efficiency perturbation (%)"),
                                  ("membrane", "Membrane-permeability perturbation (%)")]):
    d = rc[rc.kind == kind].sort_values("pct")
    axs[0].plot(d.pct, d.M14, MARK[j] + "-", color=PAL[j], mec="white", mew=1.2, label=lbl.split(" (")[0])
    axs[1].plot(d.pct, d.S_rank, MARK[j] + "-", color=PAL[j], mec="white", mew=1.2, label=lbl.split(" (")[0])
axs[0].axhline(M_TARGET, color=INK2, ls=":", lw=1.2)
axs[0].set_ylabel("14-day delivered dose (mg cm$^{-2}$)"); axs[1].set_ylabel("$S_{rank}$")
for a, l in zip(axs, "ab"):
    a.set_xlabel("Perturbation from nominal (%)")
    a.text(-0.02, 1.04, f"({l})", transform=a.transAxes, fontsize=11)
axs[1].legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
save(fig, F + "fig_robustness.png")

# ------------------------------------------------ tornado
se = pd.read_csv(R + "sensitivity.csv").sort_values("range_S").tail(12)
fig, ax = plt.subplots(figsize=(7.2, 4.8))
y = np.arange(len(se))
ax.barh(y + 0.19, se.dS_minus, color=PAL[0], height=0.36, label="−10%")
ax.barh(y - 0.19, se.dS_plus, color=PAL[1], height=0.36, label="+10%")
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(se.parameter, fontsize=9)
ax.set_xlabel("Change in $S_{rank}$ relative to the selected design")
ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
save(fig, F + "fig_tornado.png")

# ------------------------------------------------ bandage schematic (selected design, not to scale)
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.set_axis_off()
layers = [("Outer insulating shell", SEL.L_shell, "#BDBDBD"), ("Cooling layer (ice pack)", SEL.L_cool, "#9DB4D6"),
          ("Thermal-interface hydrogel", SEL.L_gel, "#C9D6E8"), ("Diclofenac hydrogel reservoir", SEL.L_res, "#8FA9C9"),
          ("Rate-controlling membrane", SEL.L_mem, "#4D4D4D"), ("Hydrated wicking liner", SEL.L_liner, "#E0E0E0")]
disp = [0.7, 1.6, 0.9, 0.8, 0.25, 0.7]
yv = 6.0
for (name, L, col), h in zip(layers, disp):
    ax.add_patch(mpatches.Rectangle((0.6, yv - h), 4.2, h, facecolor=col, edgecolor=INK2, lw=0.8))
    ax.text(5.0, yv - h / 2, f"{name} — {L*1e3:.2f} mm", va="center", fontsize=10, color=INK)
    yv -= h
ax.text(5.0, yv - 0.25, "contact interface: $R_c(p)$, $h_\\Gamma^{eff}(p)$", fontsize=9.5, color=INK2, va="center")
tissue = [("Skin — 2.0 mm (incl. 0.10 mm effective barrier)", 0.6, "#E3C2B6"), ("Subcutaneous fat — 5.0 mm", 0.7, "#E6DCB8"),
          ("Hamstring muscle — 30 mm (probe at 10 mm; 37 °C at base)", 1.0, "#C48E88")]
yv -= 0.5
for name, h, col in tissue:
    ax.add_patch(mpatches.Rectangle((0.6, yv - h), 4.2, h, facecolor=col, edgecolor=INK2, lw=0.8))
    ax.text(5.0, yv - h / 2, name, va="center", fontsize=10, color=INK)
    yv -= h
ax.annotate("", xy=(0.35, yv + 0.2), xytext=(0.35, 6.0), arrowprops=dict(arrowstyle="->", color=INK2))
ax.text(0.2, 3.2, "x", fontsize=11, color=INK2)
ax.text(0.6, 6.25, f"Selected design D2: ice, $p$ = {SEL.p:.0f} mmHg, $C_0$ = {SEL.C0:.0f} mg mL$^{{-1}}$, "
        f"3 × 60-min sessions of {SEL.tau_on:.0f}/{SEL.tau_off:.0f} min on/off, wear {SEL.t_wear:.1f} h d$^{{-1}}$",
        fontsize=9.5, color=INK)
ax.text(0.6, yv - 0.3, "Layer heights not to scale.", fontsize=8.5, color=INK2, style="italic")
ax.set_xlim(0, 11); ax.set_ylim(yv - 0.5, 6.5)
save(fig, F + "fig_bandage_selected.png")
print("figures done")
