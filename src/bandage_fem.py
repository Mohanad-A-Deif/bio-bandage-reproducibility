"""
bandage_fem.py -- One-dimensional finite-element solver for the coupled
thermal / diclofenac-transport / compression-contact model of the multilayer
cryo-compression bandage (revision R1).

Model (see manuscript, Section 4 and Appendix A):
  * Transient heat conduction in six bandage layers and three tissue layers,
    Pennes bioheat source in perfused tissue (optional cold-induced
    vasoconstriction, alpha_b = 0 in nominal runs), apparent-heat-capacity
    phase change in the cooling layer written in enthalpy form with a
    raised-cosine kernel chi(xi) = (1 + cos(pi xi))/2 on [-1, 1].
  * Pressure-dependent thermal contact resistance at the liner/skin interface,
    R_c(p) = R_ref * R_c0 * exp(-beta_c p).
  * Diclofenac diffusion through reservoir -> membrane -> liner -> effective
    skin barrier with layer-specific Arrhenius diffusivities evaluated with the
    local (computed) temperature, equilibrium partitioning at internal
    interfaces, contact-limited transfer at liner/skin with
    h_eff(p) = h_0 * eta_p(p), and a perfusion-limited sink at the barrier base.
Discretisation: linear (P1) finite elements, lumped mass, backward Euler,
Newton iterations on the enthalpy residual, tridiagonal direct solves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import numpy as np
from numba import njit

R_GAS = 8.314462618
K0 = 273.15

# --------------------------------------------------------------------------
# Nominal parameters
# --------------------------------------------------------------------------
# thermal: k [W/m/K], rho [kg/m3], c [J/kg/K]
LAYER_THERMAL = {
    "shell":  (0.22, 1050.0, 1750.0),
    "gel":    (0.58, 1000.0, 4000.0),
    "res":    (0.52, 1000.0, 3850.0),
    "mem":    (0.20, 1200.0, 1500.0),
    "liner":  (0.30, 1000.0, 3500.0),   # hydrated wicking liner
    # tissue (IT'IS database v4.1 average values)
    "skin":   (0.37, 1109.0, 3391.0),
    "fat":    (0.21, 911.0, 2348.0),
    "muscle": (0.49, 1090.0, 3421.0),
}
# perfusion [ml/min/kg] and metabolic heat [W/kg] (IT'IS v4.1)
TISSUE_PERF = {"skin": 106.0, "fat": 33.0, "muscle": 37.0}
TISSUE_QMET = {"skin": 1.65, "fat": 0.51, "muscle": 0.91}
RHO_B, C_B = 1050.0, 3617.0

# cooling media: k, rho, c_s, T_m [C], dT half-width [C], L_f [J/kg], T_init [C]
MEDIA = {
    "Ice":   dict(k=2.2, rho=920.0, c=2100.0, Tm=0.0, dTm=1.0, Lf=334e3, Tinit=-5.0),
    "PCM-A": dict(k=0.5, rho=900.0, c=2000.0, Tm=8.6, dTm=1.7, Lf=182e3, Tinit=4.0),
    "PCM-B": dict(k=0.5, rho=900.0, c=2000.0, Tm=6.0, dTm=1.5, Lf=182e3, Tinit=2.0),
}


@dataclass
class Params:
    # geometry [m]
    L_cool: float = 4.0e-3
    L_shell: float = 1.0e-3
    L_gel: float = 1.0e-3
    L_res: float = 1.0e-3
    L_mem: float = 0.10e-3
    L_liner: float = 1.0e-3
    L_sb: float = 0.10e-3      # effective skin barrier (part of skin)
    L_skin: float = 2.0e-3     # total skin incl. barrier
    L_fat: float = 5.0e-3
    L_muscle: float = 30.0e-3
    x_mus_depth: float = 10.0e-3  # muscle probe depth below fat/muscle interface
    medium: str = "PCM-A"
    # environment
    T_amb: float = 22.0
    h_ext: float = 10.0
    h_air_skin: float = 10.0
    T_core: float = 37.0
    T_a: float = 37.0
    alpha_b: float = 0.0       # cold-induced vasoconstriction coefficient [1/K]
    T_ref_b: float = 34.0
    # thermal contact (liner/skin)
    R_ref: float = 2.0e-3      # m2K/W, scale of normalised R_c0
    R_c0: float = 1.0
    beta_c: float = 0.025      # 1/mmHg
    h_attach: float = 5.0e3    # pack/shell to gel conductance when applied
    # drug
    C0: float = 10.0           # kg/m3 == mg/mL
    D_res: float = 4.7e-10
    D_mem: float = 1.0e-11
    D_liner: float = 1.0e-10
    D_sb: float = 3.0e-12
    E_res: float = 20.0e3      # J/mol
    E_mem: float = 30.0e3
    E_liner: float = 20.0e3
    E_sb: float = 60.0e3
    T_refD: float = 32.0
    K_res_mem: float = 0.71
    K_mem_liner: float = 1.0
    K_liner_skin: float = 0.19
    k_clr: float = 3.1e-6
    kappa: float = 0.0
    h0: float = 2.0e-8         # m/s, contact-limited transfer scale
    eta_min: float = 0.55
    eta_max: float = 1.0
    beta_p: float = 0.055
    contact_eff_scale: float = 1.0  # robustness perturbation of eta_p
    mem_perm_scale: float = 1.0     # robustness perturbation of membrane D
    h_pen: float = 1.0e-3           # equilibrium-interface penalty conductance
    # protocol
    p: float = 25.0            # mmHg
    t_wear: float = 8.0        # h
    n_sess: int = 3
    sess_window: float = 60.0  # min
    tau_on: float = 15.0       # min
    tau_off: float = 15.0      # min ; 0 => continuous
    cooling: bool = True
    drug: bool = True
    contact_coupling: bool = True
    thermal_coupling: bool = True   # Arrhenius with local T
    membrane: bool = True
    # numerics
    nel: Dict[str, int] = field(default_factory=lambda: dict(
        cool=20, shell=10, gel=12, res=20, mem=10, liner=12, sb=10, skin=16,
        fat=20, muscle=40))
    dt_cool: float = 5.0
    dt_wear: float = 30.0
    dt_off: float = 120.0
    horizon_h: float = 24.0


def eta_p(p, P: Params):
    e = P.eta_min + (P.eta_max - P.eta_min) * (1.0 - math.exp(-P.beta_p * p))
    return min(max(e, P.eta_min), P.eta_max)


# --------------------------------------------------------------------------
# Mesh construction
# --------------------------------------------------------------------------
def build_thermal(P: Params):
    med = MEDIA[P.medium]
    layers = [
        ("shell", P.L_shell) + LAYER_THERMAL["shell"],
        ("cool", P.L_cool, med["k"], med["rho"], med["c"]),
        ("gel", P.L_gel) + LAYER_THERMAL["gel"],
        ("res", P.L_res) + LAYER_THERMAL["res"],
        ("mem", P.L_mem) + LAYER_THERMAL["mem"],
        ("liner", P.L_liner) + LAYER_THERMAL["liner"],
        ("sb", P.L_sb) + LAYER_THERMAL["skin"],
        ("skin", P.L_skin - P.L_sb) + LAYER_THERMAL["skin"],
        ("fat", P.L_fat) + LAYER_THERMAL["fat"],
        ("muscle", P.L_muscle) + LAYER_THERMAL["muscle"],
    ]
    tissue_of = {"sb": "skin", "skin": "skin", "fat": "fat", "muscle": "muscle"}
    x_nodes: List[float] = []
    cap, lat, perf, qmet = [], [], [], []
    link_G, link_type = [], []   # type 0 element, 1 attach interface, 2 contact
    elem_layer = []
    node_of = {}
    x = 0.0
    for li, (name, L, k, rho, c) in enumerate(layers):
        ne = P.nel[name]
        dx = L / ne
        if li == 0:
            x_nodes.append(0.0); cap.append(0.0); lat.append(0.0); perf.append(0.0); qmet.append(0.0)
        elif name in ("gel", "sb"):
            # duplicate node -> interface link (attach or contact)
            x_nodes.append(x); cap.append(0.0); lat.append(0.0); perf.append(0.0); qmet.append(0.0)
            link_G.append(0.0); link_type.append(1 if name == "gel" else 2); elem_layer.append(-1)
        node_of[name + "_first"] = len(x_nodes) - 1
        for e in range(ne):
            i0 = len(x_nodes) - 1
            x += dx
            x_nodes.append(x); cap.append(0.0); lat.append(0.0); perf.append(0.0); qmet.append(0.0)
            i1 = i0 + 1
            for i in (i0, i1):
                cap[i] += 0.5 * dx * rho * c
                if name == "cool":
                    lat[i] += 0.5 * dx * rho * med["Lf"]
                if name in tissue_of:
                    t = tissue_of[name]
                    w = TISSUE_PERF[t] * 1e-6 / 60.0 * LAYER_THERMAL[t][1]
                    perf[i] += 0.5 * dx * w * RHO_B * C_B
                    qmet[i] += 0.5 * dx * TISSUE_QMET[t] * LAYER_THERMAL[t][1]
            link_G.append(k / dx); link_type.append(0); elem_layer.append(li)
        node_of[name + "_last"] = len(x_nodes) - 1
    xT = np.array(x_nodes)
    mus_top = P.L_cool + P.L_shell + P.L_gel + P.L_res + P.L_mem + P.L_liner + P.L_skin + P.L_fat
    i_mus = int(np.argmin(np.abs(xT - (mus_top + P.x_mus_depth))))
    T = dict(
        x=xT, cap=np.array(cap), lat=np.array(lat), perf=np.array(perf), qmet=np.array(qmet),
        G=np.array(link_G), ltype=np.array(link_type, dtype=np.int64), elem_layer=np.array(elem_layer),
        n_cool_last=node_of["cool_last"], i_shell_last=node_of["cool_last"],
        i_gel_first=node_of["gel_first"], i_liner_last=node_of["liner_last"],
        i_skin=node_of["sb_first"], i_mus=i_mus,
        Tm=med["Tm"], dTm=med["dTm"], Tinit=med["Tinit"],
        layers=layers,
    )
    # map thermal element index (link index) for drug layers
    T["elem_index_of_layer"] = {}
    for name in ("res", "mem", "liner", "sb"):
        li = [l[0] for l in layers].index(name)
        T["elem_index_of_layer"][name] = np.where(T["elem_layer"] == li)[0]
    return T


def build_drug(P: Params, TH):
    # layers: res, mem, liner, sb with duplicate nodes at the 3 interfaces
    spec = [("res", P.L_res, P.D_res, P.E_res), ("mem", P.L_mem, P.D_mem * P.mem_perm_scale, P.E_mem),
            ("liner", P.L_liner, P.D_liner, P.E_liner), ("sb", P.L_sb, P.D_sb, P.E_sb)]
    if not P.membrane:
        spec[1] = ("mem", P.L_mem, P.D_res, P.E_res)
    vol, Dref, Eact, dxs, link_type, therm_idx = [], [], [], [], [], []
    nodes = 0
    res_nodes = []
    for li, (name, L, D, E) in enumerate(spec):
        ne = P.nel[name]
        dx = L / ne
        if li == 0:
            vol.append(0.0); nodes = 1
        else:
            vol.append(0.0); nodes += 1
            link_type.append(li)  # 1: res|mem, 2: mem|liner, 3: liner|skin
            Dref.append(0.0); Eact.append(0.0); dxs.append(1.0); therm_idx.append(-1)
        tidx = TH["elem_index_of_layer"][name]
        for e in range(ne):
            i0 = nodes - 1
            vol.append(0.0); nodes += 1
            vol[i0] += 0.5 * dx; vol[i0 + 1] += 0.5 * dx
            if name == "res":
                res_nodes += [i0, i0 + 1]
            link_type.append(0); Dref.append(D); Eact.append(E); dxs.append(dx)
            therm_idx.append(int(tidx[e]))
    C_init = np.zeros(nodes)
    res_nodes = sorted(set(res_nodes))
    C_init[res_nodes] = P.C0 if P.drug else 0.0
    # liner/skin interface: index of first skin-barrier node
    lt = np.array(link_type, dtype=np.int64)
    i_contact_link = int(np.where(lt == 3)[0][0])
    return dict(vol=np.array(vol), Dref=np.array(Dref), Eact=np.array(Eact), dx=np.array(dxs),
                ltype=lt, tidx=np.array(therm_idx, dtype=np.int64), C_init=C_init,
                res_nodes=np.array(res_nodes), i_contact_link=i_contact_link,
                bandage_nodes=np.arange(0, i_contact_link + 1), n=nodes)


# --------------------------------------------------------------------------
# Protocol / time grid
# --------------------------------------------------------------------------
def build_schedule(P: Params):
    """Return time grid (s) and per-step flags: pack_on, worn, session, reset."""
    H = P.horizon_h * 3600.0
    wear_end = P.t_wear * 3600.0
    W = P.sess_window * 60.0
    sessions = [k * wear_end / P.n_sess for k in range(P.n_sess)]
    on_intervals = []
    for s in sessions:
        if P.tau_off <= 0:
            on_intervals.append((s, s + W))
        else:
            t = s
            while t < s + W - 1e-9:
                on_intervals.append((t, min(t + P.tau_on * 60.0, s + W)))
                t += (P.tau_on + P.tau_off) * 60.0
    # breakpoints
    bps = {0.0, H, wear_end}
    for s in sessions:
        bps.update([s, s + W, min(s + W + 1200.0, wear_end)])
    for a, b in on_intervals:
        bps.update([a, b])
    bps = sorted(b for b in bps if 0 <= b <= H)
    ts = [0.0]
    for a, b in zip(bps[:-1], bps[1:]):
        if b - a < 1e-9:
            continue
        mid = 0.5 * (a + b)
        in_cool = any(s - 1e-9 <= mid <= min(s + W + 1200.0, wear_end) for s in sessions)
        if mid > wear_end:
            dt = P.dt_off
        elif in_cool:
            dt = P.dt_cool
        else:
            dt = P.dt_wear
        n = max(1, int(math.ceil((b - a) / dt - 1e-9)))
        ts.extend(list(np.linspace(a, b, n + 1)[1:]))
    t = np.array(ts)
    tm = 0.5 * (t[1:] + t[:-1])           # step midpoints
    pack_on = np.zeros(len(tm), dtype=np.int64)
    if P.cooling:
        for a, b in on_intervals:
            pack_on[(tm > a) & (tm < b)] = 1
    worn = (tm < wear_end).astype(np.int64)
    session = np.zeros(len(tm), dtype=np.int64)
    for s in sessions:
        session[(tm > s) & (tm < s + W)] = 1
    reset = np.zeros(len(tm), dtype=np.int64)
    if P.cooling:
        for s in sessions:
            reset[int(np.argmin(np.abs(t[:-1] - s)))] = 1
    episode = np.full(len(tm), -1, dtype=np.int64)
    for j, (a, b) in enumerate(on_intervals):
        episode[(tm > a) & (tm < b)] = j
    return dict(t=t, pack_on=pack_on, worn=worn, session=session, reset=reset,
                episode=episode, n_episodes=len(on_intervals), sessions=sessions, W=W,
                wear_end=wear_end)


# --------------------------------------------------------------------------
# Numba kernels
# --------------------------------------------------------------------------
@njit(cache=True)
def _thomas(a, b, c, d):
    n = b.shape[0]
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@njit(cache=True)
def _latent_frac(T, Tm, dTm):
    xi = (T - Tm) / dTm
    if xi <= -1.0:
        return 0.0, 0.0
    if xi >= 1.0:
        return 1.0, 0.0
    f = 0.5 * (xi + 1.0 + math.sin(math.pi * xi) / math.pi)
    chi = 0.5 * (1.0 + math.cos(math.pi * xi))
    return f, chi / dTm


@njit(cache=True)
def _simulate_full(t, pack_on, worn, reset, cap, lat, perf, qmet, G, ltype, Tm, dTm, Tinit,
                   n_cool_last, i_shell_last, i_gel_first, i_skin, i_mus, T0, h_ext, h_air, T_amb,
                   T_core, T_a, alpha_b, T_ref_b, h_att, hc_worn,
                   dvol, dDref, dEact, ddx, dltype, dtidx, C0vec, K1, K2, K3, h_pen, heff_worn,
                   k_clr, kappa, T_refD, thermal_coupling):
    """Same as _simulate but also records muscle-probe temperature."""
    nT = cap.shape[0]; nD = dvol.shape[0]; nS = t.shape[0] - 1
    T = T0.copy(); C = C0vec.copy()
    Ts = np.empty(nS + 1); Tmu = np.empty(nS + 1); Cab = np.empty(nS + 1); Jab = np.empty(nS + 1)
    Ts[0] = T[i_skin]; Tmu[0] = T[i_mus]; Cab[0] = C[nD - 1]; Jab[0] = k_clr * C[nD - 1]
    a = np.zeros(nT); b = np.zeros(nT); c = np.zeros(nT); r = np.zeros(nT)
    ad = np.zeros(nD); bd = np.zeros(nD); cd = np.zeros(nD); rd = np.zeros(nD)
    Gl = G.copy(); Eold = np.empty(nT)
    newton_fail = 0
    energy_err = 0.0
    for n in range(nS):
        dt = t[n + 1] - t[n]
        if reset[n] == 1:
            for i in range(0, i_shell_last + 1):
                T[i] = Tinit
        for j in range(Gl.shape[0]):
            if ltype[j] == 1:
                Gl[j] = h_att if (pack_on[n] == 1 and worn[n] == 1) else 0.0
            elif ltype[j] == 2:
                Gl[j] = hc_worn if worn[n] == 1 else 0.0
        for i in range(nT):
            f, _ = _latent_frac(T[i], Tm, dTm)
            Eold[i] = cap[i] * T[i] + lat[i] * f
        mx = 0.0
        for it in range(80):
            for i in range(nT):
                f, dfdT = _latent_frac(T[i], Tm, dTm)
                r[i] = (cap[i] * T[i] + lat[i] * f - Eold[i]) / dt
                b[i] = (cap[i] + lat[i] * dfdT) / dt
                a[i] = 0.0; c[i] = 0.0
                if perf[i] > 0.0:
                    s = 1.0 + alpha_b * (T[i] - T_ref_b)
                    if s > 0.0:
                        r[i] += perf[i] * s * (T[i] - T_a)
                        b[i] += perf[i] * (s + alpha_b * (T[i] - T_a))
                    r[i] -= qmet[i]
            for j in range(Gl.shape[0]):
                g = Gl[j]
                q = g * (T[j] - T[j + 1])
                r[j] += q; r[j + 1] -= q
                b[j] += g; b[j + 1] += g
                c[j] -= g; a[j + 1] -= g
            r[0] += h_ext * (T[0] - T_amb); b[0] += h_ext
            if worn[n] == 1 and pack_on[n] == 0:
                r[i_gel_first] += h_ext * (T[i_gel_first] - T_amb); b[i_gel_first] += h_ext
            if worn[n] == 0:
                r[i_skin] += h_air * (T[i_skin] - T_amb); b[i_skin] += h_air
            r[nT - 1] = T[nT - 1] - T_core; b[nT - 1] = 1.0; a[nT - 1] = 0.0
            dT = _thomas(a, b, c, -r)
            mx = 0.0
            for i in range(nT):
                d = dT[i]
                if d > 1.0:
                    d = 1.0
                elif d < -1.0:
                    d = -1.0
                T[i] += d
                if abs(dT[i]) > mx:
                    mx = abs(dT[i])
            if mx < 1e-9:
                break
        if mx >= 1e-6:
            newton_fail += 1
        for i in range(nD):
            bd[i] = dvol[i] / dt + kappa * dvol[i]
            ad[i] = 0.0; cd[i] = 0.0
            rd[i] = dvol[i] / dt * C[i]
        for j in range(dltype.shape[0]):
            lt = dltype[j]
            if lt == 0:
                if thermal_coupling:
                    k = dtidx[j]
                    Te = 0.5 * (T[k] + T[k + 1]) + K0
                else:
                    Te = T_refD + K0
                D = dDref[j] * math.exp(-dEact[j] / R_GAS * (1.0 / Te - 1.0 / (T_refD + K0)))
                g = D / ddx[j]
                bd[j] += g; bd[j + 1] += g; cd[j] -= g; ad[j + 1] -= g
            else:
                if lt == 1:
                    h = h_pen; K = K1
                elif lt == 2:
                    h = h_pen; K = K2
                else:
                    h = heff_worn if worn[n] == 1 else 0.0; K = K3
                bd[j] += h; cd[j] -= h / K
                bd[j + 1] += h / K; ad[j + 1] -= h
        bd[nD - 1] += k_clr
        C = _thomas(ad, bd, cd, rd)
        Ts[n + 1] = T[i_skin]; Tmu[n + 1] = T[i_mus]
        Cab[n + 1] = C[nD - 1]; Jab[n + 1] = k_clr * C[nD - 1]
    return Ts, Tmu, Cab, Jab, newton_fail, T, C


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def initial_temperature(P: Params, TH):
    """Steady state with bandage worn, pack detached (one very long implicit step)."""
    n = len(TH["x"])
    t = np.array([0.0, 1e8])
    T0 = np.full(n, 34.0)
    dummy = build_drug(P, TH)
    hc = 1.0 / (P.R_ref * P.R_c0 * math.exp(-P.beta_c * P.p if P.contact_coupling else 0.0))
    out = _simulate_full(t, np.array([0]), np.array([1]), np.array([0]), TH["cap"], TH["lat"] * 0.0,
                         TH["perf"], TH["qmet"], TH["G"], TH["ltype"], TH["Tm"], TH["dTm"], TH["Tinit"],
                         TH["n_cool_last"], TH["i_shell_last"], TH["i_gel_first"], TH["i_skin"], TH["i_mus"],
                         T0, P.h_ext, P.h_air_skin, P.T_amb, P.T_core, P.T_a, P.alpha_b, P.T_ref_b,
                         P.h_attach, hc, dummy["vol"], dummy["Dref"], dummy["Eact"], dummy["dx"],
                         dummy["ltype"], dummy["tidx"], dummy["C_init"] * 0.0, P.K_res_mem,
                         P.K_mem_liner, P.K_liner_skin, P.h_pen, 0.0, P.k_clr, P.kappa, P.T_refD, True)
    return out[5]


def run(P: Params, return_series: bool = True):
    TH = build_thermal(P)
    DR = build_drug(P, TH)
    S = build_schedule(P)
    T0 = initial_temperature(P, TH)
    pe = P.p if P.contact_coupling else 0.0
    hc = 1.0 / (P.R_ref * P.R_c0 * math.exp(-P.beta_c * pe))
    heff = P.h0 * eta_p(pe, P) * P.contact_eff_scale
    if not P.contact_coupling:
        heff = P.h0 * P.eta_min * P.contact_eff_scale
    Ts, Tmu, Cab, Jab, nf, Tend, Cend = _simulate_full(
        S["t"], S["pack_on"], S["worn"], S["reset"], TH["cap"], TH["lat"], TH["perf"], TH["qmet"],
        TH["G"], TH["ltype"], TH["Tm"], TH["dTm"], TH["Tinit"], TH["n_cool_last"], TH["i_shell_last"],
        TH["i_gel_first"], TH["i_skin"], TH["i_mus"], T0, P.h_ext, P.h_air_skin, P.T_amb, P.T_core,
        P.T_a, P.alpha_b, P.T_ref_b, P.h_attach, hc,
        DR["vol"], DR["Dref"], DR["Eact"], DR["dx"], DR["ltype"], DR["tidx"], DR["C_init"],
        P.K_res_mem, P.K_mem_liner, P.K_liner_skin, P.h_pen, heff, P.k_clr, P.kappa, P.T_refD,
        P.thermal_coupling)
    return dict(t=S["t"], Ts=Ts, Tmu=Tmu, Cabs=Cab, J=Jab, S=S, newton_fail=nf,
                C_end=Cend, T_end=Tend, TH=TH, DR=DR, P=P)
