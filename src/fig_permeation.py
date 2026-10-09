"""Fig. 7 (isothermal cumulative permeation) and Fig. 8 (Arrhenius comparison with Caserta et al. 2024)."""
import math
from dataclasses import replace
import numpy as np, pandas as pd
import bandage_fem as bf
from bandage_fem import Params, R_GAS, K0
from figs_common import *

style()
OUT = "../figures/"


def iso_params(T_c, **kw):
    """Isothermal permeation (drug only, no cooling, bandage worn 24 h, diffusivities at T_c)."""
    P = Params(cooling=False, thermal_coupling=False, t_wear=24.0, **kw)
    f = lambda E: math.exp(-E / R_GAS * (1 / (T_c + K0) - 1 / (P.T_refD + K0)))
    return replace(P, D_res=P.D_res * f(P.E_res), D_mem=P.D_mem * f(P.E_mem), D_liner=P.D_liner * f(P.E_liner),
                   D_sb=P.D_sb * f(P.E_sb))


rows = []
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for j, Tc in enumerate([15.0, 22.0, 32.0]):
    o = bf.run(iso_params(Tc))
    t = o["t"] / 3600
    M = np.concatenate([[0], np.cumsum(0.5 * (o["J"][1:] + o["J"][:-1]) * np.diff(o["t"]))]) * 100 * 1000  # ug/cm2
    m = t <= 24
    ax.plot(t[m], M[m], color=PAL[j], label=f"{Tc:.0f} °C")
    k = [np.argmin(abs(t - h)) for h in (6, 12, 18, 24)]
    ax.plot(t[k], M[k], MARK[j], color=PAL[j], ms=7, mec="white", mew=1.5)
    for h in (2, 4, 6, 8, 12, 24):
        i = np.argmin(abs(t - h))
        rows.append(dict(T_C=Tc, t_h=h, cum_ug_cm2=M[i], flux_ug_cm2_h=o["J"][i] * 100 * 1000 * 3600))
ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative permeated diclofenac (µg cm$^{-2}$)")
ax.set_xlim(0, 24); ax.set_ylim(bottom=0)
ax.legend(title="Isothermal temperature", loc="upper left")
save(fig, OUT + "fig_permeation_isothermal.png")
pd.DataFrame(rows).to_csv("../results/permeation_isothermal.csv", index=False)

# ---------------- Arrhenius comparison ----------------
Ts = np.array([28, 32, 35, 38, 41, 43, 45.0])
Jm = []
for Tc in Ts:
    o = bf.run(iso_params(Tc))
    i = np.argmin(abs(o["t"] - 12 * 3600))
    Jm.append(o["J"][i])
Jm = np.array(Jm)
invT = 1000 / (Ts + K0)
slope = np.polyfit(1 / (Ts + K0), np.log(Jm), 1)[0]
Ea_model = -slope * R_GAS / 1000
# Caserta et al. 2024, Eur J Pharm Sci 203:106933 (PBS vehicle, mean flux at 32 and 45 C)
cas = {"Human scrotal skin": (99.5, 272.9), "Human abdominal skin": (33.3, 187.0), "Porcine skin": (56.6, 210.5)}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(invT, np.log(Jm / Jm[1]), "-", color=PAL[0], label=f"Model, coupled stack (apparent $E_a$ = {Ea_model:.1f} kJ mol$^{{-1}}$)")
ax.plot(invT, np.log(Jm / Jm[1]), "o", color=PAL[0], mec="white", mew=1.5)
x2 = 1000 / (np.array([32, 45]) + K0)
cas_rows = []
for j, (lab, (a, b)) in enumerate(cas.items()):
    Ea = math.log(b / a) * R_GAS / (1 / (32 + K0) - 1 / (45 + K0)) / 1000
    ax.plot(x2, [0, math.log(b / a)], "--", color=PAL[j + 1], lw=1.6)
    ax.plot(x2, [0, math.log(b / a)], MARK[j + 1], color=PAL[j + 1], mec="white", mew=1.5,
            label=f"Caserta et al. – {lab.lower()} ($E_a$ ≈ {Ea:.0f} kJ mol$^{{-1}}$)")
    cas_rows.append(dict(tissue=lab, J32=a, J45=b, ratio=b / a, Ea_apparent_kJmol=Ea))
ax.set_xlabel("1000 / T (K$^{-1}$)"); ax.set_ylabel("ln[ J(T) / J(32 °C) ]")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5, ncol=1)
save(fig, OUT + "fig_arrhenius_caserta.png")
pd.DataFrame(cas_rows + [dict(tissue="model", ratio=Jm[-1] / Jm[1], Ea_apparent_kJmol=Ea_model)]).to_csv(
    "../results/arrhenius_comparison.csv", index=False)
print("model Ea", Ea_model, "ratio45/32", Jm[-1] / Jm[1])
print(pd.DataFrame(rows).round(3).to_string())
