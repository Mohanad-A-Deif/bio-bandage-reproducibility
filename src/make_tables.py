"""Write LaTeX tables and number macros directly from the result files (no hand transcription)."""
import json, os
import numpy as np, pandas as pd

R, T = "../results/", "../tables/"
os.makedirs(T, exist_ok=True)
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
f1 = lambda x: f"{x:.1f}"


def tab(name, header, rows, colspec, caption, label, star=False, size=r"\small", note=None):
    env = "table*" if star else "table"
    s = [rf"\begin{{{env}}}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
         r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.12}",
         r"\resizebox{\linewidth}{!}{%" if star or len(header) > 6 else "", rf"\begin{{tabular}}{{{colspec}}}",
         r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    s += [" & ".join(map(str, r)) + r" \\" for r in rows]
    s += [r"\bottomrule", r"\end{tabular}", "}" if star or len(header) > 6 else ""]
    if note:
        s.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {note}}}")
    s.append(rf"\end{{{env}}}")
    open(T + name + ".tex", "w").write("\n".join(x for x in s if x != "") + "\n")


A = json.load(open(R + "analysis_summary.json"))
V = json.load(open(R + "verification.json"))
cand = pd.read_csv(R + "candidate_designs.csv")
front = pd.read_csv(R + "eps_constraint_frontier.csv")
scen = pd.read_csv(R + "scenarios.csv")
abl = pd.read_csv(R + "ablation.csv")
rob = pd.read_csv(R + "robustness.csv")
sens = pd.read_csv(R + "sensitivity.csv")
shifts = pd.read_csv(R + "thermal_margin_shifts.csv")
par = pd.read_csv(R + "doe_feasible_pareto.csv")
doe = pd.read_csv(R + "doe_runs.csv")
conv = pd.read_csv(R + "convergence.csv")
perm = pd.read_csv(R + "permeation_isothermal.csv")
arr = pd.read_csv(R + "arrhenius_comparison.csv")
wts = pd.read_csv(R + "weight_perturbation.csv") if os.path.exists(R + "weight_perturbation.csv") else None

# ---------------- epsilon-constraint frontier
rows = [[f2(r.eps), f2(r.L_gel_mm), f2(r.L_res_mm), f1(r.C0), f3(r.J), f2(r.Ts_min), f1(r.S_T), f3(r.M14), f3(r.S_rank)]
        for _, r in front.iterrows()]
tab("tab_eps_frontier", [r"$\varepsilon$ ($^\circ$C)", r"$L_{\mathrm{gel}}$ (mm)", r"$L_{\mathrm{res}}$ (mm)",
                         r"$C_0$ (mg\,mL$^{-1}$)", r"$J^\star$", r"$\min T_{\mathrm{skin}}$ ($^\circ$C)", r"$S_T$ (\%)",
                         r"$M_{14}$ (mg\,cm$^{-2}$)", r"$S_{\mathrm{rank}}$"], rows, "ccccccccc",
    r"$\varepsilon$-constraint frontier: best objective value $J^\star$ obtained by the multistart optimiser (ice medium, six starts per level) when the skin-temperature admissibility constraint is tightened to $T_{\mathrm{skin}}(t)\ge T_{\min}+\varepsilon$. In all cases $L_{\mathrm{cool}}\approx8.0$~mm, $L_{\mathrm{mem}}=0.05$~mm, $p\approx10$~mmHg, 15/15-min duty cycle and $t_{\mathrm{wear}}\approx7.0$~h.",
    "tab:eps_frontier")

# ---------------- thermal-margin shifts
rows = [[r.perturbation.replace("alpha_b", r"$\alpha_b$").replace("R_c0", r"$R_{c,0}$").replace("h_ext", r"$h_{\mathrm{ext}}$")
         .replace("T_amb", r"$T_\infty$").replace("k_liner", r"$k_{\mathrm{liner}}$").replace("k_gel", r"$k_{\mathrm{gel}}$")
         .replace("T_init", r"$T_{\mathrm{init}}$").replace("%", r"\%").replace(" C", r"~$^\circ$C").replace("/K", r"~K$^{-1}$"),
         f2(r.Ts_min), f"{r['shift']:+.2f}"] for _, r in shifts.iterrows()]
tab("tab_margin_shifts", ["Perturbation of the $\\varepsilon=0$ design", r"$\min T_{\mathrm{skin}}$ ($^\circ$C)",
                          r"Shift ($^\circ$C)"], rows, "lcc",
    r"Shift of the minimum skin temperature of the boundary design ($\varepsilon=0$) under parametric thermal perturbations, used to size the safety margin $\varepsilon^\star$. The vasoconstriction case is a structural (model-form) change and is reported separately.",
    "tab:margin_shifts")

# ---------------- candidate designs
rows = []
for _, r in cand.iterrows():
    lab = r.label + (r"$^{\dagger}$" if r.get("selected", False) is True or str(r.get("selected")) == "True" else "")
    rows.append([lab, r.medium, f2(r.L_cool_mm), f2(r.L_gel_mm), f2(r.L_res_mm), f2(r.L_mem_mm), f1(r.C0), f1(r.p),
                 f1(r.tau_on), f2(r.t_wear), f"{r.J:.4f}", f1(r.R_B), f1(r.S_T), f1(r.S_C), f3(r.M14), f2(r.dT_peak),
                 f2(r.Ts_min), f3(r.S_rank)])
tab("tab_candidates", ["Design", "Medium", r"$L_{\mathrm{cool}}$", r"$L_{\mathrm{gel}}$", r"$L_{\mathrm{res}}$",
                       r"$L_{\mathrm{mem}}$", r"$C_0$", r"$p$", r"$\tau_{\mathrm{on}}$", r"$t_{\mathrm{wear}}$", r"$J$",
                       r"$R_B$", r"$S_T$", r"$S_C$", r"$M_{14}$", r"$\Delta T_{\mathrm{peak}}$",
                       r"$\min T_{\mathrm{skin}}$", r"$S_{\mathrm{rank}}$"], rows, "llcccccccccccccccc",
    r"Optimisation-derived designs. D1--D5: distinct optima of $J$ at the selected margin $\varepsilon^\star$, ordered by $J$; PCM rows: best constrained designs with the two PCM media ($\varepsilon=0$); D0: boundary design ($\varepsilon=0$). Units: thicknesses in mm, $C_0$ in mg\,mL$^{-1}$, $p$ in mmHg, $\tau_{\mathrm{on}}=\tau_{\mathrm{off}}$ in min, $t_{\mathrm{wear}}$ in h, $R_B,S_T,S_C$ in \%, $M_{14}$ in mg\,cm$^{-2}$, temperatures in $^\circ$C. $^{\dagger}$Selected by the lexicographic rule of Section~\ref{subsec:selection_rule}.",
    "tab:optimized_designs", star=True)

# ---------------- scenarios
nm = {"A1": "A1 full coupled", "A2": "A2 cooling + compression", "A3": "A3 drug + compression", "A4": "A4 baseline contact"}
rows = [[nm[r.label], f"{r.J:.3f}", f1(r.R_B), f1(r.S_T), f1(r.S_C), f3(r.M14), f2(r.Ts_min), f2(r.Tmu_min), f3(r.S_rank)]
        for _, r in scen.iterrows()]
tab("tab_scenarios", ["Scenario", r"$J$", r"$R_B$ (\%)", r"$S_T$ (\%)", r"$S_C$ (\%)", r"$M_{14}$ (mg\,cm$^{-2}$)",
                      r"$\min T_{\mathrm{skin}}$ ($^\circ$C)", r"$\min T_{\mathrm{muscle}}$ ($^\circ$C)", r"$S_{\mathrm{rank}}$"],
    rows, "lcccccccc",
    r"Computational scenarios evaluated with the geometry and protocol of the selected design. $R_B$ is computed relative to the fixed reference configuration $\theta_{\mathrm{ref}}$ (scenario A4 with the baseline geometry of Table~\ref{tab:layers_bounds}).",
    "tab:arm_outcomes")

# ---------------- ablation
rows = [[r.label.replace("D at T_ref", r"$D$ at $T_{\mathrm{ref}}$"), f1(r.R_B), f1(r.S_C), f3(r.M14),
         f"{100*(r.M14/abl.M14[0]-1):+.1f}", f2(r.Ts_min), f3(r.S_rank)] for _, r in abl.iterrows()]
tab("tab_ablation", ["Configuration", r"$R_B$ (\%)", r"$S_C$ (\%)", r"$M_{14}$ (mg\,cm$^{-2}$)",
                     r"$\Delta M_{14}$ vs.\ full (\%)", r"$\min T_{\mathrm{skin}}$ ($^\circ$C)", r"$S_{\mathrm{rank}}$"],
    rows, "lcccccc",
    r"Ablation of the coupled mechanisms for the selected design (same geometry and protocol). The column $\Delta M_{14}$ is the prediction change that results from omitting a mechanism.",
    "tab:ablation_results")

# ---------------- robustness
rows = [[r.label.replace("%", r"\%").replace("E_D", r"$E_D$").replace("alpha_b", r"$\alpha_b$").replace("/K", r"~K$^{-1}$"),
         f1(r.R_B), f1(r.S_T), f1(r.S_C), f3(r.M14), f2(r.dT_peak), f2(r.Ts_min), f3(r.S_rank)] for _, r in rob.iterrows()]
tab("tab_robustness", ["Case", r"$R_B$ (\%)", r"$S_T$ (\%)", r"$S_C$ (\%)", r"$M_{14}$ (mg\,cm$^{-2}$)",
                       r"$\Delta T_{\mathrm{peak}}$ ($^\circ$C)", r"$\min T_{\mathrm{skin}}$ ($^\circ$C)", r"$S_{\mathrm{rank}}$"],
    rows, "lccccccc", r"Robustness of the selected design under contact, permeability, activation-energy and vasoconstriction perturbations.",
    "tab:robustness_results")

# ---------------- sensitivity
def tex_par(s):
    return (s.replace("D_res,ref", r"$D_{\mathrm{res,ref}}$").replace("D_sb,ref", r"$D_{\mathrm{sb,ref}}$")
            .replace("k_clr", r"$k_{\mathrm{clr}}$").replace("E_sb", r"$E_{D,\mathrm{sb}}$").replace("R_c0", r"$R_{c,0}$")
            .replace("h_0", r"$h_{\Gamma,0}$").replace("k_liner", r"$k_{\mathrm{liner}}$").replace("k_skin", r"$k_{\mathrm{skin}}$")
            .replace("omega_b", r"$\omega_b$").replace("C0", r"$C_0$").replace("Compression pressure p", r"Compression pressure $p$"))
rows = [[int(r["rank"]), tex_par(r.parameter), f"{r.dS_minus:+.4f}", f"{r.dS_plus:+.4f}", f"{r.range_S:.4f}",
         f"{r.dTs_minus:+.2f} / {r.dTs_plus:+.2f}", f"{r.dM_minus:+.3f} / {r.dM_plus:+.3f}"] for _, r in sens.iterrows()]
tab("tab_sensitivity", ["Rank", "Parameter ($\\pm10\\%$)", r"$\Delta S_{\mathrm{rank}}^{-}$", r"$\Delta S_{\mathrm{rank}}^{+}$",
                        "Range", r"$\Delta\min T_{\mathrm{skin}}$ ($^\circ$C)", r"$\Delta M_{14}$ (mg\,cm$^{-2}$)"],
    rows, "clccccc", r"Local one-at-a-time sensitivity around the selected design ($\pm10\%$ perturbations), ranked by the range of $S_{\mathrm{rank}}$.",
    "tab:sensitivity_summary", star=True)

# ---------------- Pareto representatives
idx = [730, 3448, 965, 964, 851]
p3 = par[par.pareto3]
sel = par.loc[[i for i in par.index if i in idx]] if False else None
pr = []
for k, i in enumerate(idx):
    r = par.loc[i]
    pr.append([f"P{k+1}", r.medium, f1(r.L_cool_mm), f1(r.L_res_mm), f2(r.L_mem_mm), f1(r.L_gel_mm), int(r.p_mmHg),
               r.duty_cycle.replace("continuous", "cont."), f1(r.R_B), f1(r.S_T), f3(r.M14), f2(r.dT_peak), f2(r.Ts_min)])
tab("tab_pareto", ["Design", "Medium", r"$L_{\mathrm{cool}}$", r"$L_{\mathrm{res}}$", r"$L_{\mathrm{mem}}$", r"$L_{\mathrm{gel}}$",
                   r"$p$", "Duty", r"$R_B$ (\%)", r"$S_T$ (\%)", r"$M_{14}$", r"$\Delta T_{\mathrm{peak}}$", r"$\min T_{\mathrm{skin}}$"],
    pr, "llccccccccccc",
    rf"Representative members of the non-dominated set of admissible DOE runs (objectives: maximise $R_B$, maximise $S_T$, minimise $|M_{{14}}-M^\star|$). Of the {A['doe_total']} DOE runs, {A['doe_feasible']} satisfy $T_{{\mathrm{{skin}}}}\ge10\,^\circ$C and {A['pareto3']} of these are non-dominated. Thicknesses in mm, $p$ in mmHg, $M_{{14}}$ in mg\,cm$^{{-2}}$, temperatures in $^\circ$C.",
    "tab:pareto_designs", star=True)

# ---------------- DOE summary per medium
rows = []
for med in ["Ice", "PCM-A", "PCM-B"]:
    d = doe[doe.medium == med]
    rows.append([med, len(d), int((d.Ts_min >= 10).sum()), f2(d.Ts_min.min()), f2(d.Ts_min.max()), f1(d.S_T.max()),
                 f1(d.R_B.max()), f3(d.S_rank.max()), f3(d.S_rank.median())])
tab("tab_doe_summary", ["Medium", "Runs", "Admissible", r"$\min T_{\mathrm{skin}}$ range ($^\circ$C)", "", r"max $S_T$ (\%)",
                        r"max $R_B$ (\%)", r"max $S_{\mathrm{rank}}$", r"median $S_{\mathrm{rank}}$"], rows, "lcccccccc",
    r"Summary of the full-factorial DOE ($3^4\times4^2=1296$ runs per cooling medium, 3888 forward solves in total).",
    "tab:doe_summary")

# ---------------- verification
vrows = [["Steady multilayer conduction: $T_{\\mathrm{skin}}$ FEM vs.\\ series-resistance solution",
          f"{V['steady_skin_T_fem']:.4f} vs.\\ {V['steady_skin_T_analytic']:.4f} $^\\circ$C (error {V['steady_abs_err_C']:.1e} $^\\circ$C)"],
         ["Quasi-steady isothermal flux (12 h): FEM vs.\\ series-resistance estimate",
          f"{V['qs_flux_fem']:.3e} vs.\\ {V['qs_flux_series']:.3e} kg\\,m$^{{-2}}$\\,s$^{{-1}}$ ({100*V['qs_flux_rel_err']:.1f}\\%)"],
         ["Drug mass balance over 24 h ($\\kappa=0$)", f"relative error {V['mass_balance_rel_err']:.1e}"],
         ["Residual skin-barrier depot at 24 h / daily delivered mass", f"{V['residual_depot_frac_of_daily_dose']:.1e}"],
         ["Mesh refinement ($\\times1\\to\\times4$): $\\min T_{\\mathrm{skin}}$, $M_{14}$, $J$",
          f"{V['mesh_err_Ts_min']:.3f} $^\\circ$C, {100*V['mesh_err_M14']:.3f}\\%, {100*V['mesh_err_J']:.3f}\\%"],
         ["Mesh refinement: $\\min T_{\\mathrm{muscle}}$ (probe-node location)", f"{V['mesh_err_Tmu_min']:.3f} $^\\circ$C"],
         ["Time-step refinement ($\\times1\\to\\times\\tfrac14$): $\\min T_{\\mathrm{skin}}$, $M_{14}$, $J$",
          f"{V['dt_err_Ts_min']:.3f} $^\\circ$C, {100*V['dt_err_M14']:.2f}\\%, {100*V['dt_err_J']:.3f}\\%"]]
tab("tab_verification", ["Check", "Result"], vrows, "p{0.55\\linewidth}p{0.40\\linewidth}",
    "Solver verification and numerical-convergence checks (nominal ice configuration).", "tab:identified_parameters")

# ---------------- weight perturbation
if wts is not None:
    rows = [[r.case.replace("%", r"\%").replace("w_T", r"$w_T$").replace("w_C", r"$w_C$").replace("w_p,w_u", r"$w_p,w_u$")
             .replace("w_L", r"$w_L$"), f2(r.L_gel_mm), f2(r.L_res_mm), f1(r.C0), f1(r.p), f1(r.tau_on), f2(r.t_wear),
             f3(r.M14), f2(r.Ts_min), f3(r.S_rank)] for _, r in wts.iterrows()]
    tab("tab_weights", ["Weight case", r"$L_{\mathrm{gel}}$", r"$L_{\mathrm{res}}$", r"$C_0$", r"$p$", r"$\tau_{\mathrm{on}}$",
                        r"$t_{\mathrm{wear}}$", r"$M_{14}$", r"$\min T_{\mathrm{skin}}$", r"$S_{\mathrm{rank}}$"], rows,
        "lccccccccc",
        r"Local objective-weight perturbation check: for each renormalised weight vector the full constrained multistart optimisation ($\varepsilon^\star$, ice) was repeated and the lexicographic selection rule applied. In all cases $L_{\mathrm{cool}}\approx8$~mm and $L_{\mathrm{mem}}=0.05$~mm.",
        "tab:weight_sensitivity", star=True)

# ---------------- macros for inline numbers
s = cand[cand.selected.astype(str) == "True"].iloc[0]
d1 = cand[cand.label == "D1"].iloc[0]
d0 = cand[cand.label == "D0 (eps=0)"].iloc[0]
pa = cand[cand.label == "PCM-A best"].iloc[0]; pb = cand[cand.label == "PCM-B best"].iloc[0]
a1, a2, a3, a4 = [scen.iloc[i] for i in range(4)]
ab = abl.set_index("label")
mac = dict(
    SelLabel=s.label, SelLcool=f2(s.L_cool_mm), SelLgel=f2(s.L_gel_mm), SelLres=f2(s.L_res_mm), SelLmem=f2(s.L_mem_mm),
    SelCzero=f1(s.C0), SelP=f"{s.p:.0f}", SelTau=f"{s.tau_on:.0f}", SelWear=f1(s.t_wear), SelJ=f"{s.J:.4f}",
    SelRB=f1(s.R_B), SelST=f1(s.S_T), SelSC=f1(s.S_C), SelM=f3(s.M14), SelTs=f2(s.Ts_min), SelTmu=f2(s.Tmu_min),
    SelS=f3(s.S_rank), SelCabs=f2(s.Cabs_mean),
    DoneJ=f"{d1.J:.4f}", DoneS=f3(d1.S_rank), DoneM=f3(d1.M14), DoneLgel=f2(d1.L_gel_mm), DoneLres=f2(d1.L_res_mm),
    DoneCzero=f1(d1.C0), DoneST=f1(d1.S_T),
    DzeroJ=f"{d0.J:.4f}", DzeroS=f3(d0.S_rank), DzeroTs=f2(d0.Ts_min),
    PcmaS=f3(pa.S_rank), PcmbS=f3(pb.S_rank), PcmaTs=f2(pa.Ts_min), PcmbTs=f2(pb.Ts_min), PcmaRB=f1(pa.R_B), PcmbRB=f1(pb.R_B),
    AoneRB=f1(a1.R_B), AtwoRB=f1(a2.R_B), AthreeRB=f1(a3.R_B), AfourRB=f1(a4.R_B),
    AoneM=f3(a1.M14), AthreeM=f3(a3.M14), AoneS=f3(a1.S_rank), AtwoS=f3(a2.S_rank), AthreeS=f3(a3.S_rank), AfourS=f3(a4.S_rank),
    AblThermM=f"{100*(ab.loc['No thermal coupling (D at T_ref)'].M14/ab.loc['Full coupled design'].M14-1):+.1f}",
    AblMemM=f"{100*(ab.loc['No membrane regulation'].M14/ab.loc['Full coupled design'].M14-1):+.1f}",
    AblContM=f"{100*(ab.loc['No compression/contact coupling'].M14/ab.loc['Full coupled design'].M14-1):+.1f}",
    AblContTs=f"{ab.loc['No compression/contact coupling'].Ts_min - ab.loc['Full coupled design'].Ts_min:+.2f}",
    DoeTotal=str(A["doe_total"]), DoeFeas=str(A["doe_feasible"]), DoeInfeasIce=str(A["doe_infeasible_by_medium"].get("Ice", 0)),
    ParetoN=str(A["pareto3"]), EpsStar=f2(A["eps_star"]), MaxDrop=f2(A["max_Tskin_drop"]), DeltaJ=f"{100*A['delta_J']:.0f}",
    DirichletSel=f"{100*A['rank_weight_dirichlet_top_freq'][s.label]:.1f}",
    VasoTs=f2(rob[rob.label.str.startswith('Cold')].Ts_min.iloc[0]),
    EsbHighS=f3(rob[rob.label.str.startswith('Skin-barrier')].S_rank.iloc[0]),
    EsbHighSC=f1(rob[rob.label.str.startswith('Skin-barrier')].S_C.iloc[0]),
    ContTwentyM=f3(rob[rob.label == 'Contact efficiency -20%'].M14.iloc[0]),
    ContTwentyS=f3(rob[rob.label == 'Contact efficiency -20%'].S_rank.iloc[0]),
    EaModel=f1(arr[arr.tissue == "model"].Ea_apparent_kJmol.iloc[0]),
    EaScrotal=f"{arr[arr.tissue=='Human scrotal skin'].Ea_apparent_kJmol.iloc[0]:.0f}",
    EaAbd=f"{arr[arr.tissue=='Human abdominal skin'].Ea_apparent_kJmol.iloc[0]:.0f}",
    EaPig=f"{arr[arr.tissue=='Porcine skin'].Ea_apparent_kJmol.iloc[0]:.0f}",
    RatioModel=f2(arr[arr.tissue == "model"].ratio.iloc[0]),
    PermFifteen=f1(perm[(perm.T_C == 15) & (perm.t_h == 24)].cum_ug_cm2.iloc[0]),
    PermTwentyTwo=f1(perm[(perm.T_C == 22) & (perm.t_h == 24)].cum_ug_cm2.iloc[0]),
    PermThirtyTwo=f1(perm[(perm.T_C == 32) & (perm.t_h == 24)].cum_ug_cm2.iloc[0]),
    FluxThirtyTwo=f2(perm[(perm.T_C == 32) & (perm.t_h == 12)].flux_ug_cm2_h.iloc[0]),
    FluxFifteen=f2(perm[(perm.T_C == 15) & (perm.t_h == 12)].flux_ug_cm2_h.iloc[0]),
    SensTop=tex_par(sens.iloc[0].parameter), SensSecond=tex_par(sens.iloc[1].parameter), SensThird=tex_par(sens.iloc[2].parameter),
    FrontJzero=f"{front.iloc[0].J:.4f}", FrontJone=f"{front.iloc[-1].J:.4f}",
)
open(T + "numbers.tex", "w").write("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(open(T + "numbers.tex").read())
