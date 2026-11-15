import json, numpy as np
from scipy.optimize import minimize_scalar
d = json.load(open('results/aer_validation.json')); P = d['points']
def lost_aer(r): return 1 - (r['E_aer'] - r['m']/2)/(r['E_ideal'] - r['m']/2)
def fit(rows):
    f = lambda xi: np.mean([(1 - r['F']**xi - lost_aer(r))**2 for r in rows])
    return minimize_scalar(f, bounds=(0.05, 1.5), method='bounded').x
tr = [r for r in P if r['graph'] in (0,1)]; te = [r for r in P if r['graph'] == 2]
xi = fit(tr)
mae = lambda rows, xi: float(np.mean([abs(1 - r['F']**xi - lost_aer(r)) for r in rows]))
out = dict(xi_train=float(xi), mae_train_calibrated=mae(tr, xi), mae_test_calibrated=mae(te, xi),
           mae_test_uncalibrated=mae(te, 1.0), xi_all=float(fit(P)),
           bias_test_calibrated=float(np.mean([(1 - r['F']**xi) - lost_aer(r) for r in te])),
           bias_test_uncalibrated=float(np.mean([(1 - r['F']) - lost_aer(r) for r in te])),
           n_train=len(tr), n_test=len(te))
for nn in (8,10,12):
    rows=[r for r in P if r['n']==nn]; out[f'xi_n{nn}']=float(fit(rows)); out[f'mae_test_cal_n{nn}']=mae([r for r in te if r['n']==nn], xi)
    out[f'mae_test_uncal_n{nn}']=mae([r for r in te if r['n']==nn], 1.0)
for pp in (1,2):
    out[f'xi_p{pp}']=float(fit([r for r in P if r['p']==pp]))
# by noise level (calibrated, all points)
for e2 in (0.005, 0.01, 0.02, 0.04):
    rows = [r for r in P if r['e2'] == e2]
    out[f'mean_lost_aer_e2_{e2}'] = float(np.mean([lost_aer(r) for r in rows]))
    out[f'mean_lost_esp_e2_{e2}'] = float(np.mean([1 - r['F'] for r in rows]))
    out[f'mean_lost_cal_e2_{e2}'] = float(np.mean([1 - r['F']**xi for r in rows]))
json.dump(out, open('results/aer_calibration.json','w'), indent=1); print(json.dumps(out, indent=1))
