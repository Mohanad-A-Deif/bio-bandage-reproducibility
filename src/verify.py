"""Verification of the FEM solver: analytical checks, conservation, convergence."""
import json, math
from dataclasses import replace
import numpy as np, pandas as pd
import bandage_fem as bf
from bandage_fem import Params
from study import evaluate

out = {}
P0 = Params(medium="Ice")

# 1) steady-state thermal stack without perfusion/metabolism vs series-resistance solution
P = replace(P0)
TH = bf.build_thermal(P)
TH0 = dict(TH); TH0["perf"] = TH["perf"] * 0; TH0["qmet"] = TH["qmet"] * 0
DR = bf.build_drug(P, TH)
hc = 1.0 / (P.R_ref * P.R_c0 * math.exp(-P.beta_c * P.p))
res = bf._simulate_full(np.array([0.0, 1e9]), np.array([0]), np.array([1]), np.array([0]),
                        TH0["cap"], TH0["lat"] * 0, TH0["perf"], TH0["qmet"], TH0["G"], TH0["ltype"],
                        TH0["Tm"], TH0["dTm"], TH0["Tinit"], TH0["n_cool_last"], TH0["i_shell_last"],
                        TH0["i_gel_first"], TH0["i_skin"], TH0["i_mus"], np.full(len(TH0["x"]), 30.0),
                        P.h_ext, P.h_air_skin, P.T_amb, P.T_core, P.T_a, 0.0, P.T_ref_b, P.h_attach, hc,
                        DR["vol"], DR["Dref"], DR["Eact"], DR["dx"], DR["ltype"], DR["tidx"], DR["C_init"] * 0,
                        P.K_res_mem, P.K_mem_liner, P.K_liner_skin, P.h_pen, 0.0, P.k_clr, 0.0, P.T_refD, True)
T_fem = res[5][TH["i_skin"]]
LT = bf.LAYER_THERMAL
R_band = sum(L / LT[n][0] for n, L in [("gel", P.L_gel), ("res", P.L_res), ("mem", P.L_mem), ("liner", P.L_liner)])
R_tis = P.L_skin / LT["skin"][0] + P.L_fat / LT["fat"][0] + P.L_muscle / LT["muscle"][0]
Rc = 1.0 / hc
q = (P.T_core - P.T_amb) / (1 / P.h_ext + R_band + Rc + R_tis)
T_an = P.T_amb + q * (1 / P.h_ext + R_band + Rc)
out["steady_skin_T_fem"] = float(T_fem); out["steady_skin_T_analytic"] = float(T_an)
out["steady_abs_err_C"] = float(abs(T_fem - T_an))

# 2) quasi-steady drug flux (isothermal, T = T_ref) vs series-resistance formula
Pd = replace(P0, cooling=False, thermal_coupling=False, t_wear=24.0)
o = bf.run(Pd)
t = o["t"]
i6 = int(np.argmin(abs(t - 12 * 3600)))
# reservoir mean concentration at 12 h requires state: rerun up to 12 h
Pd12 = replace(Pd, horizon_h=12.0, t_wear=24.0)
o12 = bf.run(Pd12)
DR = o12["DR"]; C = o12["C_end"]
res_nodes = DR["res_nodes"]
Cr = float(np.sum(C[res_nodes] * DR["vol"][res_nodes]) / np.sum(DR["vol"][res_nodes]))
K1, K2, K3 = Pd.K_res_mem, Pd.K_mem_liner, Pd.K_liner_skin
heff = Pd.h0 * bf.eta_p(Pd.p, Pd)
Rs = K1 * K2 * K3 * (Pd.L_res / 2) / Pd.D_res + K2 * K3 * Pd.L_mem / Pd.D_mem + K3 * Pd.L_liner / Pd.D_liner \
    + K3 / heff + Pd.L_sb / Pd.D_sb + 1 / Pd.k_clr
J_an = K1 * K2 * K3 * Cr / Rs
J_fem = float(o12["J"][-1])
out["qs_flux_fem"] = J_fem; out["qs_flux_series"] = J_an; out["qs_flux_rel_err"] = abs(J_fem - J_an) / J_an

# 3) drug mass balance over 24 h (kappa = 0)
o = bf.run(P0)
DR = o["DR"]
M0 = float(np.sum(DR["C_init"] * DR["vol"]))
delivered = float(np.trapezoid(o["J"], o["t"]))
left = float(np.sum(o["C_end"] * DR["vol"]))
out["mass_balance_rel_err"] = abs(M0 - delivered - left) / M0
# residual skin-barrier depot at end of day relative to daily delivered amount
sb = np.arange(DR["i_contact_link"] + 1, DR["n"])
out["residual_depot_frac_of_daily_dose"] = float(np.sum(o["C_end"][sb] * DR["vol"][sb]) / delivered)


# 4) mesh and time-step convergence for the key outputs
def key(P):
    m = evaluate(P)
    return dict(Ts_min=m["Ts_min"], Tmu_min=m["Tmu_min"], M14=m["M14"], J=m["J"])


rows = []
for f in [0.5, 1, 2, 4]:
    nel = {k: max(2, int(round(v * f))) for k, v in P0.nel.items()}
    rows.append(dict(study="mesh", factor=f, **key(replace(P0, nel=nel))))
for f in [2, 1, 0.5, 0.25]:
    rows.append(dict(study="dt", factor=f, **key(replace(P0, dt_cool=P0.dt_cool * f, dt_wear=P0.dt_wear * f,
                                                          dt_off=P0.dt_off * f))))
conv = pd.DataFrame(rows)
conv.to_csv("../results/convergence.csv", index=False)
ref_m = conv[(conv.study == "mesh") & (conv.factor == 4)].iloc[0]
ref_t = conv[(conv.study == "dt") & (conv.factor == 0.25)].iloc[0]
nom = conv[(conv.study == "mesh") & (conv.factor == 1)].iloc[0]
for k in ["Ts_min", "Tmu_min", "M14", "J"]:
    out[f"mesh_err_{k}"] = float(abs(nom[k] - ref_m[k]) / (abs(ref_m[k]) if k in ("M14", "J") else 1))
    out[f"dt_err_{k}"] = float(abs(nom[k] - ref_t[k]) / (abs(ref_t[k]) if k in ("M14", "J") else 1))
print(conv.round(5).to_string())
print(json.dumps(out, indent=1))
json.dump(out, open("../results/verification.json", "w"), indent=1)
