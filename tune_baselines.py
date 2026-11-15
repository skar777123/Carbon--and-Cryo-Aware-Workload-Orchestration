"""Tune heuristic baselines on VALIDATION seeds only (20000+); test seeds are never touched here."""
import json
import sys
import numpy as np
from sim import *

VAL = list(range(20000, 20006))
out = {}


def run(prov, pol, cfg=None):
    cfg = cfg or Cfg()
    cs = []
    for prof in PROFILES:
        for s in VAL:
            cs.append(Episode(cfg, prof, s, prov).run(pol)["total_cost"])
    return float(np.mean(cs))


for n in (1, 2, 3):
    out[f"static{n}|CostGreedy"] = run(f"static{n}", CostGreedy())
    print("static", n, "CostGreedy", out[f"static{n}|CostGreedy"], flush=True)
for rho in (0.5, 0.7, 0.9, 1.1):
    out[f"lf{rho}|CostGreedy"] = run("lf", CostGreedy(), Cfg(rho_star=rho))
    print("lf", rho, out[f"lf{rho}|CostGreedy"], flush=True)
bestn = min((1, 2, 3), key=lambda n: out[f"static{n}|CostGreedy"])
for th in (0.0, 0.1, 0.3):
    out[f"static{bestn}|CarbonDefer{th}"] = run(f"static{bestn}", CarbonDefer(theta=th))
    print("static", bestn, "CarbonDefer", th, out[f"static{bestn}|CarbonDefer{th}"], flush=True)
json.dump(out, open("/home/claude/ecoq/results/tuning.json", "w"), indent=1)
print("best static n:", bestn)
