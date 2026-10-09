import json, numpy as np, pandas as pd
from study import *
from optimize_designs import optimise, to_params
from analysis_helpers import summarize_runs
CASES={"Baseline":(0.400,0.320,0.100,0.130,0.050),
 "Thermal-prioritized (w_T+10%)":(0.423,0.308,0.096,0.125,0.048),
 "Delivery-prioritized (w_C+10%)":(0.388,0.341,0.097,0.126,0.048),
 "Control-prioritized (w_p,w_u+10%)":(0.391,0.313,0.108,0.140,0.048),
 "Thickness-prioritized (w_L+10%)":(0.398,0.318,0.100,0.129,0.055)}
rows=[]
for name,w in CASES.items():
    ow=dict(zip(["w_T","w_C","w_p","w_u","w_L"],w))
    res=optimise(media=("Ice",),obj_w=ow,hard=True,margin=0.5)
    rows+=summarize_runs(res,name,ow)
pd.DataFrame(rows).to_csv("../results/weight_perturbation.csv",index=False)
print(pd.DataFrame(rows).round(3).to_string())
