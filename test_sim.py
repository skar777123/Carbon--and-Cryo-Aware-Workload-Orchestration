import sys, time
sys.path.insert(0, "/home/claude/ecoq")
from sim import *
import numpy as np

cfg = Cfg()
rows = [("ClassicalOnly", "classical", ClassicalOnly()),
        ("FidFirst-AllOn", "all_on", FidelityFirst()),
        ("FidFirstDL-AllOn", "all_on", FidelityFirstDL()),
        ("FidFirstDL-LF", "lf", FidelityFirstDL()),
        ("CostGreedy-AllOn", "all_on", CostGreedy()),
        ("CostGreedy-LF", "lf", CostGreedy()),
        ("CostGreedy-S1", "static1", CostGreedy()),
        ("CostGreedy-S2", "static2", CostGreedy()),
        ("CarbonDefer-LF", "lf", CarbonDefer())]
for prof in PROFILES:
    for name, prov, pol in rows:
        res = []
        for seed in range(5):
            ep = Episode(cfg, prof, seed, prov)
            res.append(ep.run(pol))
        g = lambda k: np.mean([r[k] for r in res])
        print(f"{prof:6s} {name:17s} unserved={g('unserved'):6.1f} miss={g('miss'):6.1f} "
              f"kWh={g('total_kwh'):7.0f} (fleet {g('fleet_kwh'):6.0f}) kg={g('total_kg'):6.0f} "
              f"fracQ={g('frac_q'):.2f} util={g('qpu_util'):.2f} cd={g('cooldowns'):.1f} gates={g('gates'):.1f} "
              f"cost={g('total_cost'):7.0f}")
    print()
