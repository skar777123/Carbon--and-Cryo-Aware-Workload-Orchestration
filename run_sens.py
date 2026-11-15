"""Sensitivity sweeps (heuristic policies only; 10 test seeds x 3 profiles per setting)."""
import sys
sys.path.insert(0, "/home/claude/ecoq")
import run_eval as R

R.TEST = list(range(100, 110))
NAMES = ["ClassicalOnly", "CostGreedy/AllOn", "CostGreedy/Reactive-gating", "CostGreedy/Static-1",
         "CostGreedy/Static-2", "CarbonDefer/Static-1"]
sweeps = [("pf", "p_fridge_kw", (4.0, 8.0, 12.0, 24.0, 36.0)),
          ("tc", "t_cool_h", (2.0, 6.0, 30.0, 72.0)),
          ("zeta", "zeta", (3.0, 10.0, 30.0)),
          ("kappa", "kappa", (2.0, 2.5, 3.0, 3.5)),
          ("lam", "lam0", (6.0, 8.0, 10.0))]
if __name__ == "__main__":
    for tag, key, vals in sweeps:
        for v in vals:
            R.main({key: v}, tag=f"{tag}{v:g}", include_models=False, heur_names=NAMES)
