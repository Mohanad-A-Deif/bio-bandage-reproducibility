import numpy as np
from study import *
from optimize_designs import to_params
BREF=None
def summarize_runs(res,name,ow,delta=0.02):
    global BREF
    if BREF is None: BREF=B_reference(Params())
    rows=[]
    for r in res:
        P=to_params(np.array(r["z"]),r["medium"]); m=evaluate(P,obj_w=ow); RB,S=rank_score(m,BREF)
        rows.append(dict(case=name,L_cool_mm=P.L_cool*1e3,L_gel_mm=P.L_gel*1e3,L_res_mm=P.L_res*1e3,L_mem_mm=P.L_mem*1e3,
             C0=P.C0,p=P.p,tau_on=P.tau_on,t_wear=P.t_wear,J=m["J"],S_rank=S,M14=m["M14"],Ts_min=m["Ts_min"]))
    Jb=min(r["J"] for r in rows)
    near=[r for r in rows if r["J"]<=(1+delta)*Jb]
    sel=max(near,key=lambda r:r["S_rank"])
    sel=dict(sel); sel["n_near_optimal"]=len(near)
    return [sel]
