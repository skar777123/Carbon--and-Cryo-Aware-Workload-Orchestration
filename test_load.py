import sys
sys.path.insert(0, "/home/claude/ecoq")
from sim import *
import numpy as np

for lam in (6.0, 8.0, 10.0):
    print(f"=== lam0={lam}")
    for name, prov in [("S1", "static1"), ("S2", "static2"), ("S3", "all_on")]:
        for pn, pol in [("CG", CostGreedy()), ("FFDL", FidelityFirstDL())]:
            c2 = Cfg(lam0=lam)
            res = [Episode(c2, "wind", s, prov).run(pol) for s in range(8)]
            g = lambda k: np.mean([r[k] for r in res])
            base = g("total_kg") + 0.1 * g("total_kwh")
            viol = g("miss") + g("unserved")
            print(f"  {name}-{pn:4s} jobs={g('n_jobs'):.0f} viol%={100*viol/g('n_jobs'):5.1f} kWh={g('total_kwh'):7.0f} "
                  f"kg={g('total_kg'):6.0f} fracQ={g('frac_q'):.2f} util={g('qpu_util'):.2f} "
                  + " ".join(f"cost@z{z}={base+z*viol:7.0f}" for z in (3, 10, 30)))
