"""
study.py -- post-processing metrics, computational scenarios and helpers
for the R1 computational study. All quantities are computed from the PDE
solution produced by bandage_fem.run().
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

import bandage_fem as bf
from bandage_fem import Params

# ---------------- predefined computational targets (Table 11) -------------
T_MIN, T_MAX = 10.0, 15.0          # skin window during scheduled cooling episodes [C]
TMU_LOW, TMU_HIGH = 30.0, 35.0     # muscle-probe window during cooling sessions [C]
C_LOW, C_HIGH = 1.0e-3, 1.0e-2     # absorption-side concentration window [kg/m3] (1-10 ug/mL)
DT_REF = 5.0                       # thermal penalty normaliser [C]
DC_REF = C_LOW                     # concentration penalty normaliser [kg/m3]
P_MAX = 50.0                       # mmHg
L_TOT_MAX = 18.3e-3                # m, sum of admissible upper bounds of the six bandage layers
U_REF = 1.0
DAYS = 14
M_TARGET, M_SCALE = 0.50, 0.25     # 14-day delivered-dose target and normaliser [mg/cm2]
DTPK_SCALE = 5.0                   # peak-deviation normaliser [C]

OBJ_W = dict(w_T=0.40, w_C=0.32, w_p=0.10, w_u=0.13, w_L=0.05)
RANK_W = dict(l_B=0.35, l_T=0.35, l_C=0.30, l_D=0.10, l_dT=0.10)


def _pos(x):
    return np.maximum(x, 0.0)


def evaluate(P: Params, obj_w=None, keep=False):
    o = bf.run(P)
    t = o["t"]; S = o["S"]
    dt = np.diff(t)
    Ts = 0.5 * (o["Ts"][1:] + o["Ts"][:-1])
    Tmu = 0.5 * (o["Tmu"][1:] + o["Tmu"][:-1])
    Cab = 0.5 * (o["Cabs"][1:] + o["Cabs"][:-1])
    worn = S["worn"].astype(bool)
    target = S["episode"] >= 0              # scheduled cooling episodes (protocol)
    sess = S["session"].astype(bool)
    Tf = t[-1]
    # ---- penalties ----
    phiT = (_pos(T_MIN - Ts) ** 2) * worn \
        + (_pos(Ts - T_MAX) ** 2) * target \
        + (_pos(Tmu - TMU_HIGH) ** 2 + _pos(TMU_LOW - Tmu) ** 2) * sess
    phiC = (_pos(C_LOW - Cab) ** 2 + _pos(Cab - C_HIGH) ** 2) * worn
    phiT_n = phiT / DT_REF ** 2
    phiC_n = phiC / DC_REF ** 2
    u = S["pack_on"].astype(float)
    pterm = ((P.p / P_MAX) ** 2) * worn
    uterm = (u / U_REF) ** 2
    Ls = np.array([P.L_shell, P.L_cool, P.L_gel, P.L_res, P.L_mem, P.L_liner])
    w = obj_w or OBJ_W
    J = np.sum((w["w_T"] * phiT_n + w["w_C"] * phiC_n + w["w_p"] * pterm + w["w_u"] * uterm) * dt) / Tf \
        + w["w_L"] * (np.sum(Ls) / L_TOT_MAX) ** 2
    B = np.sum((phiT_n + phiC_n + pterm + uterm) * dt) / Tf
    # ---- thermal-window compliance ----
    viol = ((Ts > T_MAX) & target) | ((Ts < T_MIN) & worn)
    T_target = np.sum(dt[target])
    V_T = np.sum(dt[viol])
    S_T = 100.0 * (1.0 - min(max(V_T / T_target, 0.0), 1.0))
    T_wear = np.sum(dt[worn])
    S_C = 100.0 * (1.0 - min(max(np.sum(np.minimum(phiC_n, 1.0)[worn] * dt[worn]) / T_wear, 0.0), 1.0))
    # ---- session peak deviation: distance of the session-minimum skin
    #      temperature from the target window, maximised over sessions ----
    dev = 0.0
    tm = 0.5 * (t[1:] + t[:-1])
    for s0 in S["sessions"]:
        m = (tm > s0) & (tm < s0 + S["W"])
        tmin = Ts[m].min()
        dev = max(dev, T_MIN - tmin, tmin - T_MAX, 0.0)
    M_day = np.trapezoid(o["J"], t) * 100.0          # kg/m2 -> mg/cm2
    M14 = DAYS * M_day
    comps = dict(cT=float(np.sum(phiT_n * dt) / Tf), cC=float(np.sum(phiC_n * dt) / Tf),
                 cp=float(np.sum(pterm * dt) / Tf), cu=float(np.sum(uterm * dt) / Tf),
                 cL=float((np.sum(Ls) / L_TOT_MAX) ** 2))
    res = dict(J=J, B=B, **comps, S_T=S_T, S_C=S_C, dT_peak=dev, M14=M14, M_day=M_day,
               Ts_min=float(o["Ts"][o["t"] < S["wear_end"]].min()),
               Tmu_min=float(o["Tmu"][o["t"] < S["wear_end"]].min()),
               Cabs_mean_wear=float(np.sum(Cab[worn] * dt[worn]) / T_wear),
               duty=float(np.sum(u * dt) / Tf), newton_fail=o["newton_fail"],
               mass_left=float(np.sum(o["C_end"] * o["DR"]["vol"])))
    if keep:
        res["series"] = o
    return res


def rank_score(m, B_ref, w=None):
    w = w or RANK_W
    RB = 100.0 * min(max((B_ref - m["B"]) / (B_ref + 1e-12), 0.0), 1.0)
    Dhat = abs(m["M14"] - M_TARGET) / M_SCALE
    dThat = m["dT_peak"] / DTPK_SCALE
    S = w["l_B"] * RB / 100 + w["l_T"] * m["S_T"] / 100 + w["l_C"] * m["S_C"] / 100 \
        - w["l_D"] * Dhat - w["l_dT"] * dThat
    return RB, S


def scenario(P: Params, name: str) -> Params:
    if name == "A1":
        return replace(P, cooling=True, drug=True, contact_coupling=True)
    if name == "A2":
        return replace(P, cooling=True, drug=False, contact_coupling=True)
    if name == "A3":
        return replace(P, cooling=False, drug=True, contact_coupling=True)
    if name == "A4":
        return replace(P, cooling=False, drug=False, contact_coupling=True)
    raise ValueError(name)


def B_reference(P: Params):
    return evaluate(scenario(P, "A4"))["B"]


def J_from_components(m, w):
    return w["w_T"] * m["cT"] + w["w_C"] * m["cC"] + w["w_p"] * m["cp"] + w["w_u"] * m["cu"] + w["w_L"] * m["cL"]
