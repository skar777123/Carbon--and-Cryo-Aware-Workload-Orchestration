"""Evaluate all methods on held-out TEST seeds (100..129) x 3 grid profiles. Episodes are paired across methods
(same carbon trace, forecast and job stream for a given (profile, seed))."""
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/home/claude/ecoq")
from sim import *
from ppo import RouterPolicy, ProvPolicy, Agent  # noqa: F401  (needed for unpickling)

TEST = list(range(100, 130))
MODEL_DIR = "/home/claude/ecoq/models"
OUT = "/home/claude/ecoq/results/test_results.jsonl"

HEUR = {
    "ClassicalOnly": ("classical", lambda: ClassicalOnly(), {}),
    "FidelityFirst/AllOn (draft baseline)": ("all_on", lambda: FidelityFirst(), {}),
    "FidelityFirst-DL/AllOn": ("all_on", lambda: FidelityFirstDL(), {}),
    "CostGreedy/AllOn": ("all_on", lambda: CostGreedy(), {}),
    "CostGreedy/Reactive-gating": ("lf", lambda: CostGreedy(), {"rho_star": 1.1}),
    "CostGreedy/Static-1": ("static1", lambda: CostGreedy(), {}),
    "CostGreedy/Static-2": ("static2", lambda: CostGreedy(), {}),
    "CarbonDefer/Static-1": ("static1", lambda: CarbonDefer(theta=0.3), {}),
}


def _heur(task):
    name, prof, seed, over = task
    prov, mk, kw = HEUR[name]
    cfg = Cfg(**{**Cfg().__dict__, **kw, **over})
    r = Episode(cfg, prof, seed, prov).run(mk())
    r.update(method=name, profile=prof, seed=seed, over=over)
    return r


def _model(task):
    path, prof, seed, over = task
    d = pickle.load(open(path, "rb"))
    cfg = Cfg(**{**d["cfg"], **over})
    rng = np.random.default_rng(0)
    router = RouterPolicy(d["router"], rng, d["fm"], True, d["allow_defer"])
    pp = ProvPolicy(d["prov"], rng, True) if d["prov_mode"] == "ppo" else None
    ep = Episode(cfg, prof, seed, d["prov_mode"])
    r = ep.run(router, pp)
    variant = os.path.basename(path).split("_" + d["prov_mode"])[0]
    r.update(method="EcoQ/" + variant, profile=prof, seed=seed, over=over, model=os.path.basename(path))
    return r


def main(overrides=None, tag="main", include_models=True, heur_names=None):
    over = overrides or {}
    tasks_h = [(n, p, s, over) for n in (heur_names or HEUR) for p in PROFILES for s in TEST]
    models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl"))) if include_models else []
    tasks_m = [(m, p, s, over) for m in models for p in PROFILES for s in TEST]
    with Pool(2) as pool:
        rh = pool.map(_heur, tasks_h, chunksize=20)
        rm = pool.map(_model, tasks_m, chunksize=10)
    with open(OUT.replace(".jsonl", f"_{tag}.jsonl"), "w") as f:
        for r in rh + rm:
            r["tag"] = tag
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rh) + len(rm), "episodes", tag)


if __name__ == "__main__":
    main()
