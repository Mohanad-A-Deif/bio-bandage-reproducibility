import json
import numpy as np
import matplotlib.patches as mp
import bandage_fem as bf
from bandage_fem import Params
from study import scenario, T_MIN, T_MAX
from figs_common import *

style()
sel = json.load(open("../results/selected_design.json"))
SEL = Params(**{k: v for k, v in sel.items() if k in Params.__dataclass_fields__})
o = bf.run(scenario(SEL, "A1"))
t = o["t"] / 3600; m = t <= 7.5

fig = plt.figure(figsize=(13.28, 5.31))
# --- panel 1: layered stack
ax = fig.add_axes([0.01, 0.05, 0.27, 0.86]); ax.set_axis_off()
ax.text(0.0, 1.0, "Multilayer bandage", fontsize=13, weight="bold", transform=ax.transAxes, va="bottom")
layers = [("Outer shell", "#BDBDBD", .07), ("Ice cooling layer", "#9DB4D6", .16), ("Thermal hydrogel", "#C9D6E8", .09),
          ("Diclofenac reservoir", "#8FA9C9", .08), ("Membrane", "#4D4D4D", .03), ("Hydrated liner", "#E0E0E0", .07),
          ("Skin", "#E3C2B6", .08), ("Fat", "#E6DCB8", .09), ("Hamstring muscle", "#C48E88", .17)]
y = 0.95
for name, col, h in layers:
    if name == "Skin":
        y -= 0.03
    ax.add_patch(mp.Rectangle((0.02, y - h), 0.45, h, facecolor=col, edgecolor=INK2, lw=0.8, transform=ax.transAxes))
    ax.text(0.5, y - h / 2, name, fontsize=10.5, va="center", transform=ax.transAxes, color=INK)
    y -= h
ax.text(0.5, 0.437, "↕ compression-modulated contact", fontsize=8.5, color=INK2, transform=ax.transAxes, va="center")
# --- panel 2: coupling chain
ax = fig.add_axes([0.30, 0.05, 0.30, 0.86]); ax.set_axis_off()
ax.text(0.0, 1.0, "Verified coupled FEM model", fontsize=13, weight="bold", transform=ax.transAxes, va="bottom")
boxes = [(0.05, 0.78, "Bioheat + ice/PCM\nphase change  →  $T(x,t)$"), (0.05, 0.52, "Arrhenius diffusivity\n$D_\\ell(T)$ in every layer"),
         (0.05, 0.26, "Diclofenac transport\n→ flux, 14-day dose"), (0.55, 0.52, "Compression $p$\n$R_c(p)$,  $h^{eff}(p)$")]
for x, yb, txt in boxes:
    ax.add_patch(mp.FancyBboxPatch((x, yb), 0.40, 0.17, boxstyle="round,pad=0.01", facecolor="#F2F2F2", edgecolor=PAL[1],
                                   lw=1.2, transform=ax.transAxes))
    ax.text(x + 0.20, yb + 0.085, txt, ha="center", va="center", fontsize=9.5, transform=ax.transAxes, color=INK)
for (x0, y0, x1, y1) in [(0.25, 0.78, 0.25, 0.69), (0.25, 0.52, 0.25, 0.43), (0.55, 0.60, 0.45, 0.85), (0.55, 0.58, 0.45, 0.34)]:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
ax.text(0.05, 0.08, "3888-run DOE  •  ε-constrained multistart optimisation\nablation  •  robustness  •  sensitivity",
        fontsize=10, transform=ax.transAxes, color=INK)
# --- panel 3: result (two stacked panels, one y-axis each)
ax = fig.add_axes([0.665, 0.53, 0.32, 0.36])
ax.axhspan(T_MIN, T_MAX, color=GRID, alpha=0.8, lw=0)
ax.plot(t[m], o["Ts"][m], color=PAL[0], lw=1.6)
ax.set_ylabel("Skin T (°C)"); ax.set_ylim(5, 36); ax.tick_params(labelbottom=False)
ax.set_title("Cooling suppresses delivery in each episode", fontsize=12, loc="left", weight="bold")
ax2 = fig.add_axes([0.665, 0.14, 0.32, 0.36], sharex=ax)
ax2.plot(t[m], o["Cabs"][m] * 1e3, color=PAL[1], lw=1.6)
ax2.set_ylabel("Conc. (µg/mL)"); ax2.set_xlabel("Time (h)"); ax2.set_ylim(0, 10)
fig.text(0.66, 0.02, f"Selected design: min skin {o['Ts'][o['t']<SEL.t_wear*3600].min():.1f} °C, 14-day dose 0.53 mg/cm²",
         fontsize=10, color=INK)
fig.savefig("../figures/graphical_abstract.png", dpi=200)
fig.savefig("../figures/graphical_abstract.pdf")
print("ok")
