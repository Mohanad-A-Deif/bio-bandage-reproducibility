"""Post-optimisation analyses for R1: epsilon-constraint selection, scenarios, ablation,
robustness, sensitivity, Pareto extraction, weight sensitivity. Writes ../results/*.csv/json."""
import json, math
from dataclasses import replace
import numpy as np, pandas as pd
import bandage_fem as bf
from bandage_fem import Params
from study import *
from optimize_designs import to_params, optimise

R = "../results/"
Bref = B_reference(Params())
out = {"B_ref": Bref}


def design_from_row(r):
    return to_params(np.array(json.loads(r["z"])), r["medium"])


def summary(P, label, **extra):
    m = evaluate(P)
    RB, S = rank_score(m, Bref)
    return dict(label=label, medium=P.medium, L_cool_mm=P.L_cool * 1e3, L_gel_mm=P.L_gel * 1e3, L_res_mm=P.L_res * 1e3,
                L_mem_mm=P.L_mem * 1e3, C0=P.C0, p=P.p, tau_on=P.tau_on, t_wear=P.t_wear, J=m["J"], R_B=RB,
                S_T=m["S_T"], S_C=m["S_C"], M14=m["M14"], dT_peak=m["dT_peak"], Ts_min=m["Ts_min"],
                Tmu_min=m["Tmu_min"], Cabs_mean=m["Cabs_mean_wear"] * 1e3, duty=m["duty"], S_rank=S, **extra)


# ---------------------------------------------------------------- 1. epsilon-constraint frontier
files = {0.0: "optimization_runs.csv", 0.25: "optimization_runs_m0.25.csv", 0.5: "optimization_runs_m0.50.csv",
         1.0: "optimization_runs_m1.00.csv"}
opt = {e: pd.read_csv(R + f) for e, f in files.items()}
front = []
best = {}
for e, df in opt.items():
    d = df[df.medium == "Ice"].sort_values("J")
    r = d.iloc[0]
    best[e] = design_from_row(r)
    front.append(summary(best[e], f"eps={e}", eps=e, n_starts=len(d), nfev=int(d.nfev.sum())))
front = pd.DataFrame(front)
front.to_csv(R + "eps_constraint_frontier.csv", index=False)

# ---------------------------------------------------------------- 2. thermal-uncertainty shifts of T_skin,min
P0 = best[0.0]
base_min = evaluate(P0)["Ts_min"]
shifts = []
TH_BASE = dict(bf.LAYER_THERMAL)
pert = [("R_c0 -10%", dict(R_c0=0.9)), ("R_c0 +10%", dict(R_c0=1.1)), ("h_ext +10%", dict(h_ext=11.0)),
        ("T_amb -2 C", dict(T_amb=20.0)), ("alpha_b = 0.02 /K", dict(alpha_b=0.02)), ("pack T_init: -10 C", None),
        ("k_liner +10%", "liner"), ("k_gel +10%", "gel"), ("skin perfusion -10%", "perf")]
for name, kw in pert:
    if isinstance(kw, dict):
        m = evaluate(replace(P0, **kw))
    elif kw is None:
        bf.MEDIA["Ice"]["Tinit"] = -10.0
        m = evaluate(P0)
        bf.MEDIA["Ice"]["Tinit"] = -5.0
    elif kw == "perf":
        old = dict(bf.TISSUE_PERF); bf.TISSUE_PERF["skin"] *= 0.9
        m = evaluate(P0); bf.TISSUE_PERF.update(old)
    else:
        k, rho, c = bf.LAYER_THERMAL[kw]; bf.LAYER_THERMAL[kw] = (1.1 * k, rho, c)
        m = evaluate(P0); bf.LAYER_THERMAL[kw] = (k, rho, c)
    shifts.append(dict(perturbation=name, Ts_min=m["Ts_min"], shift=m["Ts_min"] - base_min))
shifts = pd.DataFrame(shifts)
shifts.to_csv(R + "thermal_margin_shifts.csv", index=False)
param = shifts[~shifts.perturbation.str.startswith("alpha_b")]
max_drop = float(-param["shift"].min())
eps_star = min([e for e in sorted(files) if e >= max_drop] or [max(files)])
out.update(max_Tskin_drop=max_drop, eps_star=eps_star)

# ---------------------------------------------------------------- 3. candidate designs at eps*
d = opt[eps_star]
d = d[d.medium == "Ice"].sort_values("J")
cands, seen = [], []
DELTA_J = 0.02   # near-optimality tolerance (relative) for the lexicographic selection rule
for _, r in d.iterrows():
    P = design_from_row(r)
    key = (round(P.L_gel * 1e4), round(P.L_res * 1e4), round(P.L_mem * 1e5), round(P.C0), round(P.t_wear, 1))
    if key in seen:
        continue
    seen.append(key)
    cands.append(P)
cand_rows = [summary(P, f"D{i+1}") for i, P in enumerate(cands[:5])]
Jbest = cand_rows[0]["J"]
near = [r for r in cand_rows if r["J"] <= (1 + DELTA_J) * Jbest]
sel_label = max(near, key=lambda r: r["S_rank"])["label"]
for r in cand_rows:
    r["near_optimal"] = r["J"] <= (1 + DELTA_J) * Jbest
    r["selected"] = r["label"] == sel_label
out.update(J_best=Jbest, delta_J=DELTA_J, selected=sel_label)
# best constrained PCM designs (eps = 0) for comparison
for med in ["PCM-A", "PCM-B"]:
    r = opt[0.0][opt[0.0].medium == med].sort_values("J").iloc[0]
    cand_rows.append(summary(design_from_row(r), f"{med} best"))
# the boundary design (eps = 0)
cand_rows.append(summary(best[0.0], "D0 (eps=0)"))
cand = pd.DataFrame(cand_rows)
cand.to_csv(R + "candidate_designs.csv", index=False)
SEL = cands[int(sel_label[1:]) - 1]
json.dump({k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in SEL.__dict__.items() if k != "nel"},
          open(R + "selected_design.json", "w"), indent=1)

# ---------------------------------------------------------------- 4. scenarios
scen = [summary(scenario(SEL, s), s) for s in ["A1", "A2", "A3", "A4"]]
pd.DataFrame(scen).to_csv(R + "scenarios.csv", index=False)

# ---------------------------------------------------------------- 5. ablation
abl = [summary(SEL, "Full coupled design"),
       summary(replace(SEL, thermal_coupling=False), "No thermal coupling (D at T_ref)"),
       summary(replace(SEL, membrane=False), "No membrane regulation"),
       summary(replace(SEL, contact_coupling=False), "No compression/contact coupling"),
       summary(replace(SEL, cooling=False, contact_coupling=False), "Drug-only transport baseline")]
pd.DataFrame(abl).to_csv(R + "ablation.csv", index=False)

# ---------------------------------------------------------------- 6. robustness
rob = [summary(SEL, "Nominal design"),
       summary(replace(SEL, contact_eff_scale=0.9), "Contact efficiency -10%"),
       summary(replace(SEL, contact_eff_scale=0.8), "Contact efficiency -20%"),
       summary(replace(SEL, mem_perm_scale=1.1), "Membrane permeability +10%"),
       summary(replace(SEL, mem_perm_scale=0.9), "Membrane permeability -10%"),
       summary(replace(SEL, contact_eff_scale=0.8, mem_perm_scale=0.9), "Combined (contact -20%, permeability -10%)"),
       summary(replace(SEL, E_sb=100e3), "Skin-barrier E_D = 100 kJ/mol"),
       summary(replace(SEL, alpha_b=0.02), "Cold-induced vasoconstriction (alpha_b = 0.02 /K)")]
pd.DataFrame(rob).to_csv(R + "robustness.csv", index=False)
# robustness curves (finer)
curve = []
for s in [1.0, 0.95, 0.9, 0.85, 0.8]:
    curve.append(dict(kind="contact", pct=round((s - 1) * 100), **summary(replace(SEL, contact_eff_scale=s), "c")))
for s in [0.9, 0.95, 1.0, 1.05, 1.1]:
    curve.append(dict(kind="membrane", pct=round((s - 1) * 100), **summary(replace(SEL, mem_perm_scale=s), "m")))
pd.DataFrame(curve).to_csv(R + "robustness_curves.csv", index=False)

# ---------------------------------------------------------------- 7. local sensitivity (+-10%)
def sens_case(name, f):
    lo, hi = summary(f(0.9), name + " -10%"), summary(f(1.1), name + " +10%")
    return dict(parameter=name, dS_minus=lo["S_rank"] - nom["S_rank"], dS_plus=hi["S_rank"] - nom["S_rank"],
                dTs_minus=lo["Ts_min"] - nom["Ts_min"], dTs_plus=hi["Ts_min"] - nom["Ts_min"],
                dM_minus=lo["M14"] - nom["M14"], dM_plus=hi["M14"] - nom["M14"])


nom = summary(SEL, "nom")


def thermal_layer(layer):
    def f(s):
        k, rho, c = bf.LAYER_THERMAL[layer]
        bf.LAYER_THERMAL[layer] = (k * s, rho, c)
        return _Deferred(layer, (k, rho, c), replace(SEL))
    return f


class _Deferred:  # helper to restore after evaluation
    pass


sens = []
cases = [("Membrane permeability scaling", lambda s: replace(SEL, mem_perm_scale=s)),
         ("Cooling-episode duration", lambda s: replace(SEL, tau_on=SEL.tau_on * s, tau_off=SEL.tau_off * s)),
         ("Reservoir diffusivity D_res,ref", lambda s: replace(SEL, D_res=SEL.D_res * s)),
         ("Skin-barrier diffusivity D_sb,ref", lambda s: replace(SEL, D_sb=SEL.D_sb * s)),
         ("Cooling-layer thickness", lambda s: replace(SEL, L_cool=SEL.L_cool * s)),
         ("Thermal-gel thickness", lambda s: replace(SEL, L_gel=SEL.L_gel * s)),
         ("Reservoir thickness", lambda s: replace(SEL, L_res=SEL.L_res * s)),
         ("Initial loading C0", lambda s: replace(SEL, C0=SEL.C0 * s)),
         ("Effective clearance k_clr", lambda s: replace(SEL, k_clr=SEL.k_clr * s)),
         ("Daily wear time", lambda s: replace(SEL, t_wear=SEL.t_wear * s)),
         ("Skin-barrier activation energy E_sb", lambda s: replace(SEL, E_sb=SEL.E_sb * s)),
         ("Contact resistance R_c0", lambda s: replace(SEL, R_c0=SEL.R_c0 * s)),
         ("Contact-transfer scale h_0", lambda s: replace(SEL, h0=SEL.h0 * s)),
         ("Compression pressure p", lambda s: replace(SEL, p=SEL.p * s))]
for name, f in cases:
    sens.append(sens_case(name, f))
# thermal-property cases need module-level edits
for name, layer in [("Liner conductivity k_liner", "liner"), ("Skin conductivity k_skin", "skin")]:
    k, rho, c = bf.LAYER_THERMAL[layer]
    res = {}
    for s in (0.9, 1.1):
        bf.LAYER_THERMAL[layer] = (k * s, rho, c)
        res[s] = summary(SEL, name)
    bf.LAYER_THERMAL[layer] = (k, rho, c)
    sens.append(dict(parameter=name, dS_minus=res[0.9]["S_rank"] - nom["S_rank"], dS_plus=res[1.1]["S_rank"] - nom["S_rank"],
                     dTs_minus=res[0.9]["Ts_min"] - nom["Ts_min"], dTs_plus=res[1.1]["Ts_min"] - nom["Ts_min"],
                     dM_minus=res[0.9]["M14"] - nom["M14"], dM_plus=res[1.1]["M14"] - nom["M14"]))
old = dict(bf.TISSUE_PERF); res = {}
for s in (0.9, 1.1):
    bf.TISSUE_PERF.update({k: v * s for k, v in old.items()})
    res[s] = summary(SEL, "perf")
    bf.TISSUE_PERF.update(old)
sens.append(dict(parameter="Tissue perfusion omega_b", dS_minus=res[0.9]["S_rank"] - nom["S_rank"],
                 dS_plus=res[1.1]["S_rank"] - nom["S_rank"], dTs_minus=res[0.9]["Ts_min"] - nom["Ts_min"],
                 dTs_plus=res[1.1]["Ts_min"] - nom["Ts_min"], dM_minus=res[0.9]["M14"] - nom["M14"],
                 dM_plus=res[1.1]["M14"] - nom["M14"]))
sens = pd.DataFrame(sens)
sens["range_S"] = (sens.dS_plus - sens.dS_minus).abs()
sens = sens.sort_values("range_S", ascending=False)
sens["rank"] = np.arange(1, len(sens) + 1)
sens.to_csv(R + "sensitivity.csv", index=False)

# ---------------------------------------------------------------- 8. DOE Pareto (feasible runs)
doe = pd.read_csv(R + "doe_runs.csv")
feas = doe[doe.Ts_min >= T_MIN].copy()
feas["dose_dev"] = (feas.M14 - M_TARGET).abs()
F = np.c_[-feas.R_B.values, -feas.S_T.values, feas.dose_dev.values]
nd = np.ones(len(F), bool)
for i in range(len(F)):
    if nd[i]:
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            nd[i] = False
feas["pareto3"] = nd
F2 = np.c_[-feas.R_B.values, -feas.S_T.values]
nd2 = np.array([not (np.all(F2 <= F2[i], axis=1) & np.any(F2 < F2[i], axis=1)).any() for i in range(len(F2))])
feas["pareto2"] = nd2
feas.to_csv(R + "doe_feasible_pareto.csv", index=False)
out.update(doe_total=len(doe), doe_feasible=int(len(feas)), pareto3=int(nd.sum()), pareto2=int(nd2.sum()),
           doe_feasible_by_medium=feas.medium.value_counts().to_dict(),
           doe_infeasible_by_medium=doe[doe.Ts_min < T_MIN].medium.value_counts().to_dict())

# ---------------------------------------------------------------- 9. ranking-weight sensitivity (Dirichlet)
rng = np.random.default_rng(2026)
pool = cand.iloc[:5]
wins = np.zeros(len(pool))
for _ in range(5000):
    a = rng.dirichlet([1, 1, 1]); b = rng.uniform(0, 0.2, 2)
    w = dict(l_B=a[0], l_T=a[1], l_C=a[2], l_D=b[0], l_dT=b[1])
    sc = []
    for _, r in pool.iterrows():
        Dh = abs(r.M14 - M_TARGET) / M_SCALE
        sc.append(w["l_B"] * r.R_B / 100 + w["l_T"] * r.S_T / 100 + w["l_C"] * r.S_C / 100 - w["l_D"] * Dh
                  - w["l_dT"] * r.dT_peak / DTPK_SCALE)
    wins[int(np.argmax(sc))] += 1
out["rank_weight_dirichlet_top_freq"] = dict(zip(pool.label, (wins / wins.sum()).round(3).tolist()))
json.dump(out, open(R + "analysis_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
print(front.round(3).to_string()); print(shifts.round(3).to_string()); print(cand.round(3).to_string())
print(pd.DataFrame(scen).round(3).to_string()); print(pd.DataFrame(abl).round(3).to_string())
print(pd.DataFrame(rob).round(3).to_string()); print(sens.round(4).to_string())
