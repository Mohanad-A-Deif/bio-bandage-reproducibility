"""Multistart bounded optimisation of J(theta) seeded from the best DOE designs."""
import json, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import minimize
from study import *

NAMES = ["L_cool", "L_gel", "L_res", "L_mem", "C0", "p", "tau_on", "t_wear"]
LB = np.array([2e-3, 0.5e-3, 0.5e-3, 0.05e-3, 5.0, 10.0, 10.0, 7.0])
UB = np.array([8e-3, 3.0e-3, 3.0e-3, 0.30e-3, 20.0, 40.0, 30.0, 9.0])


def to_params(z, medium, base=None, **kw):
    x = LB + np.clip(z, 0, 1) * (UB - LB)
    base = base or Params()
    return replace(base, medium=medium, L_cool=x[0], L_gel=x[1], L_res=x[2], L_mem=x[3], C0=x[4], p=x[5],
                   tau_on=x[6], tau_off=x[6], t_wear=x[7], **kw)


def make_obj(medium, obj_w=None, hard=True, margin=0.0):
    """J(theta); with hard=True the admissibility constraint T_skin >= T_MIN
    (Table 11) is enforced through an exact (non-smooth) exterior penalty."""
    cache = {}

    def f(z):
        key = tuple(np.round(z, 6))
        if key not in cache:
            m = evaluate(to_params(z, medium), obj_w=obj_w)
            lim = T_MIN + margin
            cache[key] = m["J"] + (1.0 + 10.0 * max(0.0, lim - m["Ts_min"]) if (hard and m["Ts_min"] < lim) else 0.0)
        return cache[key]
    return f, cache


def run_start(args):
    medium, z0, obj_w, hard, margin = args
    f, cache = make_obj(medium, obj_w, hard, margin)
    r = minimize(f, z0, method="Powell", bounds=[(0, 1)] * len(NAMES),
                 options=dict(xtol=1e-3, ftol=1e-6, maxfev=1500))
    return dict(medium=medium, z=r.x.tolist(), J=float(r.fun), nfev=len(cache), z0=list(z0))


def seeds_from_doe(doe, medium, n):
    d = doe[doe.medium == medium].sort_values("J").head(n)
    out = []
    for _, r in d.iterrows():
        on = {"10/10": 10, "15/15": 15, "20/20": 20, "continuous": 20}[r.duty_cycle]
        x = np.array([r.L_cool_mm * 1e-3, r.L_gel_mm * 1e-3, r.L_res_mm * 1e-3, r.L_mem_mm * 1e-3, 10.0,
                      r.p_mmHg, on, 8.0])
        out.append((x - LB) / (UB - LB))
    return out


def optimise(n_starts=6, media=("Ice", "PCM-A", "PCM-B"), obj_w=None, hard=True, margin=0.0):
    doe = pd.read_csv("../results/doe_runs.csv")
    if hard:
        doe = doe[doe.Ts_min >= T_MIN + margin]
    jobs = [(m, z0, obj_w, hard, margin) for m in media for z0 in seeds_from_doe(doe, m, n_starts)]
    with Pool(2) as pool:
        res = pool.map(run_start, jobs, chunksize=1)
    return res


if __name__ == "__main__":
    t0 = time.time()
    import sys
    hard = "--soft" not in sys.argv
    margin = float(sys.argv[sys.argv.index("--margin") + 1]) if "--margin" in sys.argv else 0.0
    media = ("Ice",) if margin > 0 else ("Ice", "PCM-A", "PCM-B")
    res = optimise(hard=hard, margin=margin, media=media)
    Bref = B_reference(Params())
    rows = []
    for r in res:
        P = to_params(np.array(r["z"]), r["medium"])
        m = evaluate(P)
        RB, S = rank_score(m, Bref)
        rows.append(dict(medium=r["medium"], z0=json.dumps(r["z0"]), L_cool_mm=P.L_cool * 1e3, L_gel_mm=P.L_gel * 1e3,
                         L_res_mm=P.L_res * 1e3, L_mem_mm=P.L_mem * 1e3, C0=P.C0, p=P.p, tau_on=P.tau_on,
                         t_wear=P.t_wear, nfev=r["nfev"], R_B=RB, S_rank=S, z=json.dumps(r["z"]),
                         **{k: v for k, v in m.items()}))
    df = pd.DataFrame(rows).sort_values("J")
    df["hard_constraint"] = hard
    df["margin"] = margin
    df.to_csv("../results/optimization_runs%s%s.csv" % ("" if hard else "_soft", "" if margin == 0 else "_m%.2f" % margin), index=False)
    print("time %.0f s, total evals %d" % (time.time() - t0, df.nfev.sum()))
    print(df[["medium", "L_cool_mm", "L_gel_mm", "L_res_mm", "L_mem_mm", "C0", "p", "tau_on", "t_wear", "J",
              "R_B", "S_T", "S_C", "M14", "dT_peak", "Ts_min", "Tmu_min", "S_rank"]].round(3).to_string())
