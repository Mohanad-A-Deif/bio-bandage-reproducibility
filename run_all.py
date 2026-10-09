"""Regenerate every result, table and figure of the R1 manuscript from scratch.
Usage:  python run_all.py      (about 25-30 min on two CPU cores)"""
import os, subprocess, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
for d in ("../results", "../figures", "../tables"):
    os.makedirs(d, exist_ok=True)
steps = [["verify.py"], ["doe.py"], ["optimize_designs.py"], ["optimize_designs.py", "--margin", "0.25"],
         ["optimize_designs.py", "--margin", "0.5"], ["optimize_designs.py", "--margin", "1.0"],
         ["analysis.py"], ["weights.py"], ["fig_permeation.py"], ["figures.py"], ["graphical_abstract.py"],
         ["make_tables.py"]]
for s in steps:
    print(">>", " ".join(s), flush=True)
    subprocess.run([sys.executable] + s, check=True)
print("done: results/, figures/, tables/")
