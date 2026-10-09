import itertools, time, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from dataclasses import replace
from study import *
LEV=dict(L_cool=[2e-3,4e-3,8e-3],L_res=[0.5e-3,1e-3,2e-3],L_mem=[0.05e-3,0.1e-3,0.3e-3],
         L_gel=[0.5e-3,1e-3,2e-3],p=[10,20,30,40],duty=["10/10","15/15","20/20","continuous"])
DUTY={"10/10":(10,10),"15/15":(15,15),"20/20":(20,20),"continuous":(20,0)}
def one(args):
    med,lc,lr,lm,lg,p,d=args
    on,off=DUTY[d]
    P=Params(medium=med,L_cool=lc,L_res=lr,L_mem=lm,L_gel=lg,p=p,tau_on=on,tau_off=off)
    m=evaluate(P)
    return dict(medium=med,L_cool_mm=lc*1e3,L_res_mm=lr*1e3,L_mem_mm=lm*1e3,L_gel_mm=lg*1e3,p_mmHg=p,duty_cycle=d,**m)
if __name__=="__main__":
    jobs=[(med,)+c for med in ["Ice","PCM-A","PCM-B"] for c in itertools.product(*LEV.values())]
    t0=time.time()
    with Pool(2) as pool: rows=pool.map(one,jobs,chunksize=16)
    df=pd.DataFrame(rows)
    Bref=B_reference(Params())
    sc=[rank_score(r,Bref) for r in rows]
    df["R_B"]=[a for a,b in sc]; df["S_rank"]=[b for a,b in sc]
    df.to_csv("../results/doe_runs.csv",index=False)
    print(len(df),"runs in %.1f s"%(time.time()-t0), "newton fails", df.newton_fail.sum())
    print(df.groupby("medium")[["S_rank","R_B","S_T","S_C","M14","dT_peak","Ts_min"]].describe().T.round(3))
