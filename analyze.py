"""Turn raw result CSV/JSON into LaTeX tables, PDF figures and number macros (paper/numbers.tex)."""
import sys, json, os, time
sys.path.insert(0, '/home/claude/ecoq')
import numpy as np, pandas as pd, scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import env as E

OUT = '/home/claude/paper'
os.makedirs(f'{OUT}/tables', exist_ok=True); os.makedirs(f'{OUT}/figs', exist_ok=True)
rng = np.random.default_rng(0)
NUM = {}
# Okabe-Ito colour-blind safe palette
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', verm='#D55E00', sky='#56B4E9', purple='#CC79A7', yellow='#F0E442', grey='#666666')
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'pdf.fonttype': 42, 'font.family': 'serif'})


def fmt(x, d=1):
    return f'{x:,.{d}f}'


def boot_ci(v, n=4000):
    """95% CI of the mean. If v is a Series indexed by (theta, lam, regime, seed) a two-level (cluster) bootstrap is used:
    scenarios are resampled with replacement, then episodes within each resampled scenario."""
    if isinstance(v, pd.Series) and isinstance(v.index, pd.MultiIndex) and 'theta' in v.index.names:
        groups = [g.values for _, g in v.groupby(level=['theta', 'lam', 'regime'])]
        m = []
        for _ in range(n):
            pick = rng.integers(0, len(groups), len(groups))
            m.append(np.mean(np.concatenate([groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in pick])))
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    v = np.asarray(v); m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def setnum(k, v):
    NUM[k] = v


# ------------------------------------------------------------------ load
H = pd.read_csv('results/main_heur.csv')
R = pd.read_csv('results/main_rl.csv') if os.path.exists('results/main_rl.csv') else None
KEY = ['theta', 'lam', 'regime', 'seed']
if R is not None:
    R['rl_seed'] = R.policy.str.extract(r'_s(\d)')[0].astype(int)
    R['kind'] = np.where(R.policy.str.startswith('PPO-BC'), 'PPO-BC', 'PPO-scratch')
    Ragg = R.groupby(['kind'] + KEY).mean(numeric_only=True).reset_index().rename(columns={'kind': 'policy'})
    ALL = pd.concat([H, Ragg], ignore_index=True)
else:
    ALL = H.copy()
METR = ['E_dyn', 'C_dyn', 'slo_miss', 'rework', 'util_qpu', 'offload', 'cost', 'defer', 'lat_h', 'E_total', 'C_total', 'mit', 'drain_h']


def wide(metric, sub=None):
    d = ALL if sub is None else sub
    return d.pivot_table(index=KEY, columns='policy', values=metric)


def paired(pol, ref, metric, mask=None, kind='pct'):
    """per-episode paired change of pol relative to ref (negative = reduction)."""
    w = wide(metric)
    if mask is not None:
        w = w[mask(w.index.to_frame())]
    a, b = w[pol], w[ref]
    r = 100 * (a / b - 1) if kind == 'pct' else (a - b)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = boot_ci(r)
    try:
        p = st.wilcoxon(a - b).pvalue
    except Exception:
        p = np.nan
    return float(r.mean()), lo, hi, p, len(r)


active = lambda idx: idx['theta'] > 0
hard = lambda idx: idx['theta'] >= 0.35

# =========================================================== Table 1 headline
order = ['Classical-only', 'Fidelity-first', 'Speed-first', 'Carbon-shift', 'Energy-greedy', 'EcoQ', 'EcoQ-Oracle']
if R is not None:
    order += ['PPO-BC', 'PPO-scratch']
names = {'EcoQ': 'EcoQ (ours)', 'EcoQ-Oracle': 'EcoQ, perfect information', 'PPO-BC': 'PPO, planner-initialised',
         'PPO-scratch': 'PPO, from scratch', 'Fidelity-first': 'Fidelity-first offload', 'Speed-first': 'Speed-first offload',
         'Carbon-shift': 'Carbon-shift [Wiesner-style]', 'Energy-greedy': 'Energy-greedy'}
rows = []
sub = ALL[ALL.theta > 0]
for p in order:
    s = sub[sub.policy == p]
    dE = paired(p, 'Classical-only', 'E_dyn', active) if p != 'Classical-only' else (0, 0, 0, np.nan, 0)
    dC = paired(p, 'Classical-only', 'C_dyn', active) if p != 'Classical-only' else (0, 0, 0, np.nan, 0)
    rows.append(dict(p=p, dE=dE, dC=dC, slo=100 * s.slo_miss.mean(), rw=100 * s.rework.mean(), util=100 * s.util_qpu.mean(),
                     off=100 * s.offload.mean(), Ed=s.E_dyn.mean(), Cd=s.C_dyn.mean()))
with open(f'{OUT}/tables/tab_main.tex', 'w') as f:
    f.write('\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n')
    f.write('Policy & $\\Delta E_{\\mathrm{dyn}}$ (\\%) & $\\Delta C_{\\mathrm{dyn}}$ (\\%) & SLO miss (\\%) & Rework (\\%) & QPU util. (\\%) & Offload (\\%)\\\\\\midrule\n')
    for r in rows:
        def cell(t):
            if r['p'] == 'Classical-only':
                return '--'
            m, lo, hi, p, n = t
            if abs(m) > 500:
                return f'{m:+,.0f}$^{{\\dagger}}$'
            return f'{m:+.1f} [{lo:+.1f}, {hi:+.1f}]'
        f.write(f"{names.get(r['p'], r['p'])} & {cell(r['dE'])} & {cell(r['dC'])} & {r['slo']:.1f} & {r['rw']:.1f} & {r['util']:.1f} & {r['off']:.1f}\\\\\n")
        if r['p'] in ('Classical-only', 'Energy-greedy'):
            f.write('\\midrule\n') if r['p'] == 'Energy-greedy' else None
    f.write('\\bottomrule\\end{tabular}\n')
for r in rows:
    k = r['p'].replace('-', '').replace(' ', '')
    for nm, t in (('dE', r['dE']), ('dC', r['dC'])):
        setnum(f'{k}{nm}', f'{t[0]:+.1f}'); setnum(f'{k}{nm}lo', f'{t[1]:+.1f}'); setnum(f'{k}{nm}hi', f'{t[2]:+.1f}')
    setnum(f'{k}SLO', f"{r['slo']:.1f}"); setnum(f'{k}RW', f"{r['rw']:.1f}"); setnum(f'{k}UTIL', f"{r['util']:.1f}"); setnum(f'{k}OFF', f"{r['off']:.1f}")

# EcoQ vs Speed-first, vs Energy-greedy, vs Carbon-shift paired
for ref in ('Speed-first', 'Carbon-shift', 'Energy-greedy'):
    for met, nm in (('E_dyn', 'E'), ('C_dyn', 'C'), ('cost', 'Cost')):
        m, lo, hi, p, n = paired('EcoQ', ref, met, active)
        k = ref.replace('-', '')
        setnum(f'EcoQvs{k}{nm}', f'{m:+.1f}'); setnum(f'EcoQvs{k}{nm}lo', f'{lo:+.1f}'); setnum(f'EcoQvs{k}{nm}hi', f'{hi:+.1f}')
        setnum(f'EcoQvs{k}{nm}p', f'{p:.1e}')
m, lo, hi, p, n = paired('EcoQ', 'Classical-only', 'cost', active)
setnum('EcoQvsClassicalCost', f'{m:+.1f}'); setnum('EcoQvsClassicalCostlo', f'{lo:+.1f}'); setnum('EcoQvsClassicalCosthi', f'{hi:+.1f}')
setnum('nEpisodesActive', int(n)); setnum('EcoQvsClassicalEp', f'{paired("EcoQ","Classical-only","E_dyn",active)[3]:.1e}')
setnum('EcoQvsClassicalCp', f'{paired("EcoQ","Classical-only","C_dyn",active)[3]:.1e}')
# hard cells only
for met, nm in (('E_dyn', 'E'), ('C_dyn', 'C')):
    m, lo, hi, p, n = paired('EcoQ', 'Classical-only', met, hard)
    setnum(f'EcoQvsClassicalHard{nm}', f'{m:+.1f}'); setnum(f'EcoQvsClassicalHard{nm}lo', f'{lo:+.1f}'); setnum(f'EcoQvsClassicalHard{nm}hi', f'{hi:+.1f}')
# oracle gap
m, lo, hi, p, n = paired('EcoQ-Oracle', 'EcoQ', 'cost', active)
setnum('OracleGapCost', f'{m:+.1f}'); setnum('OracleGapCostlo', f'{lo:+.1f}'); setnum('OracleGapCosthi', f'{hi:+.1f}'); setnum('OracleGapCostp', f'{p:.1e}')
# theta = 0 no-regret check
s0 = ALL[ALL.theta == 0]
w0 = s0.pivot_table(index=KEY, columns='policy', values='E_dyn')
setnum('NoRegretEcoQOff', f"{100*s0[s0.policy=='EcoQ'].offload.mean():.2f}")
setnum('NoRegretEdiff', f"{100*(w0['EcoQ']/w0['Classical-only']-1).mean():+.2f}")
setnum('NoRegretFidEdyn', f"{(w0['Fidelity-first']/w0['Classical-only']).mean():,.0f}")
wc0 = s0.pivot_table(index=KEY, columns='policy', values='C_dyn')
setnum('NoRegretCdiff', f"{100*(wc0['EcoQ']/wc0['Classical-only']-1).mean():+.2f}")
setnum('NoRegretSLOEcoQ', f"{100*s0[s0.policy=='EcoQ'].slo_miss.mean():.2f}")

# ============================================================== Table 2 cells
rows = []
for th in (0.0, 0.2, 0.35, 0.5):
    for lam in (2.0, 4.0, 6.0):
        mk = lambda idx, th=th, lam=lam: (idx['theta'] == th) & (idx['lam'] == lam)
        c = ALL[(ALL.theta == th) & (ALL.lam == lam)]
        cl, ec, sp = [c[c.policy == p] for p in ('Classical-only', 'EcoQ', 'Speed-first')]
        dE = paired('EcoQ', 'Classical-only', 'E_dyn', mk)[0] if th > 0 else float('nan')
        dC = paired('EcoQ', 'Classical-only', 'C_dyn', mk)[0] if th > 0 else float('nan')
        dEs = paired('EcoQ', 'Speed-first', 'E_dyn', mk)[0] if th > 0 else float('nan')
        rows.append((th, lam, cl.E_dyn.mean(), ec.E_dyn.mean(), dE, dC, 100 * cl.slo_miss.mean(), 100 * sp.slo_miss.mean(), 100 * ec.slo_miss.mean(),
                     100 * ec.offload.mean(), 100 * ec.util_qpu.mean(), len(ec)))
with open(f'{OUT}/tables/tab_cells.tex', 'w') as f:
    f.write('\\begin{tabular}{@{}rrrrrrrrrr@{}}\\toprule\n')
    f.write('$\\theta$ & $\\lambda$ & $E^{\\mathrm{CL}}_{\\mathrm{dyn}}$ & $E^{\\mathrm{EcoQ}}_{\\mathrm{dyn}}$ & $\\Delta E$ (\\%) & $\\Delta C$ (\\%) & SLO$^{\\mathrm{CL}}$ & SLO$^{\\mathrm{SF}}$ & SLO$^{\\mathrm{EcoQ}}$ & Offl. (\\%)\\\\\\midrule\n')
    last = None
    for r in rows:
        if last is not None and r[0] != last:
            f.write('\\midrule\n')
        last = r[0]
        dE = '--' if np.isnan(r[4]) else f'{r[4]:+.1f}'
        dC = '--' if np.isnan(r[5]) else f'{r[5]:+.1f}'
        f.write(f'{r[0]:.2f} & {r[1]:.0f} & {r[2]:,.0f} & {r[3]:,.0f} & {dE} & {dC} & {r[6]:.1f} & {r[7]:.1f} & {r[8]:.1f} & {r[9]:.1f}\\\\\n')
    f.write('\\bottomrule\\end{tabular}\n')
best_cell = max([r for r in rows if r[0] > 0], key=lambda r: -r[4])
setnum('BestCellTheta', f'{best_cell[0]:.2f}'); setnum('BestCellLam', f'{best_cell[1]:.0f}'); setnum('BestCellDE', f'{best_cell[4]:+.1f}'); setnum('BestCellDC', f'{best_cell[5]:+.1f}')
setnum('BestCellSLOcl', f'{best_cell[6]:.1f}'); setnum('BestCellSLOeco', f'{best_cell[8]:.1f}')

# =========================================== Heatmap figure (delta E, C, SLO)
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.1))
ths, lams = [0.2, 0.35, 0.5], [2.0, 4.0, 6.0]
for a, (met, ttl) in zip(ax, (('E_dyn', r'$\Delta E_{\rm dyn}$ (%)'), ('C_dyn', r'$\Delta C_{\rm dyn}$ (%)'), ('slo_miss', r'$\Delta$ SLO miss (pp)'))):
    M = np.zeros((len(ths), len(lams)))
    for i, th in enumerate(ths):
        for j, lam in enumerate(lams):
            mk = lambda idx, th=th, lam=lam: (idx['theta'] == th) & (idx['lam'] == lam)
            M[i, j] = paired('EcoQ', 'Classical-only', met, mk, kind='pct' if met != 'slo_miss' else 'abs')[0] * (100 if met == 'slo_miss' else 1)
    v = np.abs(M).max()
    im = a.imshow(M, cmap='BrBG_r', vmin=-v, vmax=v, aspect='auto')
    a.set_xticks(range(3)); a.set_xticklabels([f'{l:.0f}' for l in lams]); a.set_yticks(range(3)); a.set_yticklabels([f'{t}' for t in ths])
    a.set_xlabel(r'arrival rate $\lambda$ (jobs/h)'); a.set_title(ttl)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center', fontsize=7, color=('white' if abs(M[i,j]) > 0.55*v else 'black'))
    for s_ in a.spines.values():
        s_.set_visible(False)
ax[0].set_ylabel(r'classical hardness $\theta$')
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_heat.pdf'); plt.close()

# ===================================================== Fixed-load / break-even
P_fixQ = sum(q['P_fix'] for q in E.QPUS) + sum(q['P_dyn'] for q in E.QPUS) * (E.DEFAULTS['cal_dur'] / 86400.0)  # W
pue, T = E.DEFAULTS['pue'], E.DEFAULTS['horizon_h']
EfixQ = P_fixQ * pue * T / 1000.0  # kWh / week
setnum('EfixQ', f'{EfixQ:,.0f}'); setnum('PfixQ', f'{P_fixQ/1000:.1f}')
rows = []
for th in (0.2, 0.35, 0.5):
    for lam in (2.0, 4.0, 6.0):
        c = ALL[(ALL.theta == th) & (ALL.lam == lam)]
        cl, ec = c[c.policy == 'Classical-only'].E_dyn.mean(), c[c.policy == 'EcoQ'].E_dyn.mean()
        d = cl - ec
        Cl, Ce = c[c.policy == 'Classical-only'].C_dyn.mean(), c[c.policy == 'EcoQ'].C_dyn.mean()
        efixtot = c[c.policy == 'EcoQ'].E_total.mean()
        rows.append((th, lam, d, d / (pue * T) , EfixQ / max(d, 1e-9), 100 * EfixQ / efixtot, Cl - Ce, efixtot, cl))
with open(f'{OUT}/tables/tab_breakeven.tex', 'w') as f:
    f.write('\\begin{tabular}{@{}rrrrrr@{}}\\toprule\n')
    f.write('$\\theta$ & $\\lambda$ & $E^{\\mathrm{CL}}_{\\mathrm{dyn}}-E^{\\mathrm{EcoQ}}_{\\mathrm{dyn}}$ (kWh/wk) & $P^{\\mathrm{BE}}_{\\mathrm{fix}}$ (kW) & Workload $\\times$ for break-even & Fixed share of $E_{\\mathrm{total}}$ (\\%)\\\\\\midrule\n')
    for r in rows:
        mult = '$>10^{4}$' if r[4] > 1e4 or r[2] <= 0 else f'{r[4]:,.0f}'
        f.write(f'{r[0]:.2f} & {r[1]:.0f} & {r[2]:,.0f} & {r[3]:.2f} & {mult} & {100*EfixQ/r[7]:.0f}\\\\\n')
    f.write('\\bottomrule\\end{tabular}\n')
setnum('MaxDeltaE', f'{max(r[2] for r in rows):,.0f}'); setnum('MaxPBE', f'{max(r[3] for r in rows):.1f}')
setnum('MinBEmult', f'{min(r[4] for r in rows):,.1f}')
setnum('QfixShareMin', f'{min(100*EfixQ/r[7] for r in rows):.0f}'); setnum('QfixShareMax', f'{max(100*EfixQ/r[7] for r in rows):.0f}')
# whole-facility change EcoQ vs classical (total energy incl. fixed)
mT = paired('EcoQ', 'Classical-only', 'E_total', active)
setnum('TotalEchange', f'{mT[0]:+.2f}'); mTc = paired('EcoQ', 'Classical-only', 'C_total', active); setnum('TotalCchange', f'{mTc[0]:+.2f}')

# figure: stacked energy, hardest cell
c = ALL[(ALL.theta == 0.5) & (ALL.lam == 6.0)]
clE, ecE = c[c.policy == 'Classical-only'].E_dyn.mean(), c[c.policy == 'EcoQ'].E_dyn.mean()
Ecls_idle = E.DEFAULTS['n_nodes'] * E.DEFAULTS['P_idle'] * pue * T / 1000
fig, ax = plt.subplots(figsize=(3.4, 2.3))
labs = ['Classical-only\nfacility\n(no QPUs)', 'Hybrid facility\nEcoQ\nscheduling']
seg = [(Ecls_idle, 0), (0, EfixQ)]
dyn = [clE, ecE]
x = np.arange(2)
ax.bar(x, [Ecls_idle, Ecls_idle], color=C['grey'], label='classical idle load')
ax.bar(x, [0, EfixQ], bottom=[Ecls_idle, Ecls_idle], color=C['blue'], label='QPU fixed load (cryo + control)')
ax.bar(x, dyn, bottom=[Ecls_idle, Ecls_idle + EfixQ], color=C['orange'], label='dynamic (policy-attributable)')
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('energy per week (kWh)')
ax.legend(frameon=False, loc='upper left', fontsize=6.5)
for i, d in enumerate(dyn):
    ax.text(i, Ecls_idle + (EfixQ if i else 0) + d + 150, f'{d:,.0f}', ha='center', fontsize=7)
ax.set_ylim(0, (Ecls_idle + EfixQ + ecE) * 1.6)
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_fixed.pdf'); plt.close()
setnum('HybridTotal', f'{Ecls_idle+EfixQ+ecE:,.0f}'); setnum('NoQPUTotal', f'{Ecls_idle+clE:,.0f}')
setnum('HybridRatio', f'{(Ecls_idle+EfixQ+ecE)/(Ecls_idle+clE):.1f}')
setnum('HardClE', f'{clE:,.0f}'); setnum('HardEcE', f'{ecE:,.0f}'); setnum('HardIdle', f'{Ecls_idle:,.0f}')

# ======================================================== Ablation
if os.path.exists('results/ablation.csv'):
    A = pd.read_csv('results/ablation.csv'); Ao = pd.read_csv('results/ablation_oracle.csv'); Ao['variant'] = 'perfect information (upper bound)'
    A = pd.concat([A, Ao], ignore_index=True)
    # common-weight objective (nominal alpha=beta=.5, lam=2, delta=1; latency-fraction term omitted), so that variants with
    # modified weights remain comparable
    A['score'] = 0.5 * A.E_dyn / 1.0 + 0.5 * A.C_dyn * 1000 / 300.0 + 2.0 * A.slo_miss * A.n_jobs + 1.0 * A.rework * A.n_jobs
    K2 = ['theta', 'lam', 'regime', 'seed']
    full = A[A.variant == 'EcoQ (full)'].set_index(K2)
    rows = []
    for v in list(dict.fromkeys(A.variant)):
        d = A[A.variant == v].set_index(K2)
        j = d.join(full, rsuffix='_f')
        pc = lambda m: (100 * (j[m] / j[m + '_f'] - 1)).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append((v, pc('score').mean(), boot_ci(pc('score')), pc('E_dyn').mean(), pc('C_dyn').mean(), 100 * d.slo_miss.mean(), 100 * d.rework.mean(), d.defer.mean() * 100,
                     st.wilcoxon(j['score'] - j['score_f']).pvalue if v != 'EcoQ (full)' else np.nan))
    with open(f'{OUT}/tables/tab_ablation.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n')
        f.write('Variant & $\\Delta$ objective (\\%) & $\\Delta E_{\\mathrm{dyn}}$ (\\%) & $\\Delta C_{\\mathrm{dyn}}$ (\\%) & SLO miss (\\%) & Rework (\\%) & Deferred (\\%)\\\\\\midrule\n')
        for r in rows:
            if r[0] == 'EcoQ (full)':
                f.write(f'EcoQ (full) & -- & -- & -- & {r[5]:.1f} & {r[6]:.1f} & {r[7]:.1f}\\\\\\midrule\n')
            else:
                f.write(f"{r[0].replace('w/o', 'w/o')} & {r[1]:+.1f} [{r[2][0]:+.1f}, {r[2][1]:+.1f}] & {r[3]:+.1f} & {r[4]:+.1f} & {r[5]:.1f} & {r[6]:.1f} & {r[7]:.1f}\\\\\n")
        f.write('\\bottomrule\\end{tabular}\n')
    for r in rows:
        k = {'w/o carbon term': 'AblCarbon', 'w/o energy term': 'AblEnergy', 'w/o temporal shifting': 'AblShift', 'w/o mitigation choice': 'AblMit',
             'w/o fidelity shield': 'AblShield', 'w/o set-up awareness': 'AblSetup', 'w/o SLO/latency terms': 'AblSLO',
             'perfect information (upper bound)': 'AblOracle'}.get(r[0])
        if k:
            setnum(k + 'Cost', f'{r[1]:+.1f}'); setnum(k + 'E', f'{r[3]:+.1f}'); setnum(k + 'C', f'{r[4]:+.1f}'); setnum(k + 'SLO', f'{r[5]:.1f}'); setnum(k + 'RW', f'{r[6]:.1f}')
            setnum(k + 'p', f'{r[8]:.1e}' if not np.isnan(r[8]) else 'n/a')
        else:
            setnum('AblFullSLO', f'{r[5]:.1f}'); setnum('AblFullRW', f'{r[6]:.1f}'); setnum('AblFullDefer', f'{r[7]:.1f}')

# ======================================================== Sensitivity
if os.path.exists('results/sensitivity.csv'):
    S = pd.read_csv('results/sensitivity.csv')
    rows = []
    for k in list(dict.fromkeys(S.setting)):
        d = S[S.setting == k]
        w = d.pivot_table(index=['regime', 'seed'], columns='policy', values=['E_dyn', 'C_dyn', 'slo_miss', 'cost', 'offload'])
        dEc = 100 * (w['E_dyn']['EcoQ'] / w['E_dyn']['Classical-only'] - 1).mean()
        dCc = 100 * (w['C_dyn']['EcoQ'] / w['C_dyn']['Classical-only'] - 1).mean()
        dEs = 100 * (w['E_dyn']['EcoQ'] / w['E_dyn']['Speed-first'] - 1).mean()
        dCs = 100 * (w['C_dyn']['EcoQ'] / w['C_dyn']['Speed-first'] - 1).mean()
        dK = 100 * (w['cost']['EcoQ'] / w['cost']['Classical-only'] - 1).mean()
        dKs = 100 * (w['cost']['EcoQ'] / w['cost']['Speed-first'] - 1).mean()
        rows.append((k, dEc, dCc, dEs, dCs, dK, dKs, 100 * w['slo_miss']['EcoQ'].mean(), 100 * w['offload']['EcoQ'].mean()))
    with open(f'{OUT}/tables/tab_sens.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n')
        f.write('Setting & \\multicolumn{2}{c}{vs.\\ Classical-only (\\%)} & \\multicolumn{2}{c}{vs.\\ Speed-first (\\%)} & \\multicolumn{2}{c}{Cost vs.\\ CL / SF (\\%)} & Offl. (\\%)\\\\\n')
        f.write('& $\\Delta E$ & $\\Delta C$ & $\\Delta E$ & $\\Delta C$ & CL & SF & \\\\\\midrule\n')
        for r in rows:
            lab = (r[0].replace('xi=Aer-calibrated (optimistic)', r'$\xi$ = Aer-calibrated (optimistic)').replace('rho=0.3', r'$\rho{=}0.3$').replace('eta=1.0', r'$\eta{=}1.0$')
                   .replace('sigma_k=0.5', r'$\sigma_\kappa{=}0.5$').replace(' x4', r' $\times4$').replace(' x3', r' $\times3$')
                   .replace('set-up 0 s', 'set-up 0\\,s').replace('set-up 1800 s', 'set-up 1800\\,s'))
            f.write(f'{lab} & {r[1]:+.1f} & {r[2]:+.1f} & {r[3]:+.1f} & {r[4]:+.1f} & {r[5]:+.1f} & {r[6]:+.1f} & {r[8]:.1f}\\\\\n')
        f.write('\\bottomrule\\end{tabular}\n')
    setnum('SensMinCostVsCL', f'{min(r[5] for r in rows):+.1f}'); setnum('SensMaxCostVsCL', f'{max(r[5] for r in rows):+.1f}')
    setnum('SensMinCostVsSF', f'{min(r[6] for r in rows):+.1f}'); setnum('SensMaxCostVsSF', f'{max(r[6] for r in rows):+.1f}')
    for r in rows:
        if r[0].startswith('xi=Aer'):
            setnum('SensXiOffload', f'{r[8]:.1f}'); setnum('SensXiCostCL', f'{r[5]:+.1f}'); setnum('SensXiCostSF', f'{r[6]:+.1f}')
        if r[0] == 'nominal':
            setnum('SensNomOffload', f'{r[8]:.1f}'); setnum('SensNomCostCL', f'{r[5]:+.1f}'); setnum('SensNomE', f'{r[1]:+.1f}'); setnum('SensNomC', f'{r[2]:+.1f}')

# ======================================================== Pareto
if os.path.exists('results/pareto.csv'):
    Pp = pd.read_csv('results/pareto.csv')
    g = Pp.groupby('setting').mean(numeric_only=True)
    ref = g.loc['Classical-only']
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    eco = ['E-only', 'E-heavy', 'balanced', 'C-heavy', 'C-only']
    xs = [100 * g.loc[k, 'C_dyn'] / ref.C_dyn for k in eco]; ys = [100 * g.loc[k, 'E_dyn'] / ref.E_dyn for k in eco]
    ax.plot(xs, ys, '-o', color=C['blue'], ms=4, lw=1, label=r'EcoQ, $\alpha{:}\beta$ sweep')
    for k, mk_, col in (('Classical-only', 's', C['grey']), ('Speed-first', '^', C['orange']), ('Carbon-shift', 'v', C['green']), ('Energy-greedy', 'D', C['purple'])):
        ax.scatter(100 * g.loc[k, 'C_dyn'] / ref.C_dyn, 100 * g.loc[k, 'E_dyn'] / ref.E_dyn, marker=mk_, color=col, s=22, label=k, zorder=3)
    for i_, k in enumerate(('C-only, low latency weight', 'balanced, high latency weight')):
        ax.scatter(100 * g.loc[k, 'C_dyn'] / ref.C_dyn, 100 * g.loc[k, 'E_dyn'] / ref.E_dyn, marker='x', color=C['verm'], s=20, zorder=3,
                   label='EcoQ, latency-weight variants' if i_ == 0 else None)
    ax.set_xlabel(r'$C_{\rm dyn}$ index (Classical-only = 100)'); ax.set_ylabel(r'$E_{\rm dyn}$ index (Classical-only = 100)')
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_pareto.pdf'); plt.close()
    with open(f'{OUT}/tables/tab_pareto.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}lrrrr@{}}\\toprule\nWeights / policy & $E_{\\mathrm{dyn}}$ & $C_{\\mathrm{dyn}}$ & SLO miss (\\%) & Defer (\\%)\\\\\\midrule\n')
        for k in eco + ['C-only, low latency weight', 'balanced, high latency weight', 'Classical-only', 'Speed-first', 'Carbon-shift', 'Energy-greedy']:
            f.write(f"{k} & {100*g.loc[k,'E_dyn']/ref.E_dyn:.1f} & {100*g.loc[k,'C_dyn']/ref.C_dyn:.1f} & {100*g.loc[k,'slo_miss']:.1f} & {100*g.loc[k,'defer']:.1f}\\\\\n")
        f.write('\\bottomrule\\end{tabular}\n')
    setnum('ParetoConly', f"{100*g.loc['C-only','C_dyn']/ref.C_dyn-100:+.1f}"); setnum('ParetoConlyE', f"{100*g.loc['C-only','E_dyn']/ref.E_dyn-100:+.1f}")
    setnum('ParetoEonly', f"{100*g.loc['E-only','E_dyn']/ref.E_dyn-100:+.1f}"); setnum('ParetoEonlyC', f"{100*g.loc['E-only','C_dyn']/ref.C_dyn-100:+.1f}")
    setnum('ParetoConlySLO', f"{100*g.loc['C-only','slo_miss']:.1f}"); setnum('ParetoEonlySLO', f"{100*g.loc['E-only','slo_miss']:.1f}")

# ======================================================== RL table + curves
if R is not None:
    rows = []
    sc = ALL[ALL.theta > 0]
    ec = sc[sc.policy == 'EcoQ'].set_index(KEY)
    for kind in ('PPO-BC', 'PPO-scratch'):
        sub = R[(R.kind == kind) & (R.theta > 0)]
        per_seed = []
        for sd in sorted(sub.rl_seed.unique()):
            d = sub[sub.rl_seed == sd].set_index(KEY)
            j = d.join(ec, rsuffix='_e')
            per_seed.append((100 * (j.cost / j.cost_e - 1).mean(), 100 * (j.E_dyn / j.E_dyn_e - 1).mean(), 100 * (j.C_dyn / j.C_dyn_e - 1).mean(),
                             100 * d.slo_miss.mean(), 100 * d.rework.mean(), st.wilcoxon(j.cost - j.cost_e).pvalue))
        a = np.array(per_seed)
        rows.append((kind, a[:, 0].mean(), a[:, 0].std(ddof=1) if len(a) > 1 else 0, a[:, 1].mean(), a[:, 1].std(ddof=1) if len(a) > 1 else 0,
                     a[:, 2].mean(), a[:, 2].std(ddof=1) if len(a) > 1 else 0, a[:, 3].mean(), a[:, 4].mean(), a[:, 5].max(), len(a)))
    with open(f'{OUT}/tables/tab_rl.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\nLearned policy ($n$ seeds) & Cost vs.\\ EcoQ (\\%) & $E_{\\mathrm{dyn}}$ vs.\\ EcoQ (\\%) & $C_{\\mathrm{dyn}}$ vs.\\ EcoQ (\\%) & SLO miss (\\%) & Rework (\\%) & max $p$ (Wilcoxon)\\\\\\midrule\n')
        for r in rows:
            f.write(f"{names[r[0]]} ({r[10]}) & {r[1]:+.2f} $\\pm$ {r[2]:.2f} & {r[3]:+.2f} $\\pm$ {r[4]:.2f} & {r[5]:+.2f} $\\pm$ {r[6]:.2f} & {r[7]:.1f} & {r[8]:.1f} & {r[9]:.1e}\\\\\n")
        f.write('\\bottomrule\\end{tabular}\n')
    for r in rows:
        k = r[0].replace('-', '')
        setnum(k + 'CostVsEcoQ', f'{r[1]:+.2f}'); setnum(k + 'CostSd', f'{r[2]:.2f}'); setnum(k + 'EVsEcoQ', f'{r[3]:+.2f}'); setnum(k + 'CVsEcoQ', f'{r[5]:+.2f}')
        setnum(k + 'Pmax', f'{r[9]:.1e}'); setnum(k + 'N', r[10])
    # curves
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    import glob
    for fn in sorted(glob.glob('results/train_*_s*.json')):
        d = json.load(open(fn)); cur = [c_ for c_ in d['curve'] if 'val_ratio' in c_]
        nm = os.path.basename(fn)[6:-5]
        col = C['blue'] if nm.startswith('bc') else C['verm']
        ax.plot([c_['steps'] / 1e3 for c_ in cur], [c_['val_ratio'] for c_ in cur], color=col, lw=1, alpha=.85,
                label=('PPO, planner-initialised' if nm == 'bc_s0' else ('PPO, from scratch' if nm == 'scratch_s0' else None)))
    ax.axhline(1.0, color='k', lw=.8, ls='--', label='EcoQ planner (= 1)')
    ax.set_xlabel('training steps (thousands)'); ax.set_ylabel('validation cost / planner cost'); ax.legend(frameon=False)
    ax.set_ylim(0.95, min(ax.get_ylim()[1], 2.0))
    plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_learn.pdf'); plt.close()
    sj = {}
    for fn in sorted(glob.glob('results/train_*_s*.json')):
        d = json.load(open(fn)); sj[os.path.basename(fn)[6:-5]] = d
    setnum('BCacc', f"{100*np.mean([d['bc_acc'] for k, d in sj.items() if k.startswith('bc')]):.1f}")
    setnum('BCvalbest', f"{np.mean([d['best_val'] for k, d in sj.items() if k.startswith('bc')]):.3f}")
    setnum('ScratchValbest', f"{np.mean([d['best_val'] for k, d in sj.items() if k.startswith('scratch')]):.3f}")
    setnum('TrainTimeMin', f"{np.mean([d['time'] for d in sj.values()])/60:.1f}")

# ======================================================== Aer validation
A_ = json.load(open('results/aer_validation.json')); Cal = json.load(open('results/aer_calibration.json'))
P = A_['points']
la = np.array([1 - (r['E_aer'] - r['m'] / 2) / (r['E_ideal'] - r['m'] / 2) for r in P]); Fs = np.array([r['F'] for r in P]); ns = np.array([r['n'] for r in P])
xi = Cal['xi_train']
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.2))
for a, (pred, ttl) in zip(ax[:2], ((1 - Fs, r'ESP model ($\xi=1$)'), (1 - Fs ** xi, rf'calibrated ($\xi={xi:.2f}$)'))):
    for n_, col in ((8, C['blue']), (10, C['orange']), (12, C['verm'])):
        a.scatter(la[ns == n_], pred[ns == n_], s=8, color=col, label=f'n={n_}')
    a.plot([0, 1], [0, 1], 'k--', lw=.7); a.set_xlabel('lost fraction, Qiskit Aer'); a.set_ylabel('lost fraction, model'); a.set_title(ttl); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
ax[0].legend(frameon=False, fontsize=6)
xs_ = [8, 10, 12]; ys_ = [Cal['xi_n8'], Cal['xi_n10'], Cal['xi_n12']]
fit_ = 0.342 * (8.0 / np.array(xs_)) ** 0.82
ax[2].plot(xs_, ys_, 'o', color=C['blue'], label='fitted per n'); ax[2].plot(xs_, fit_, '-', color=C['verm'], lw=1, label=r'$0.342\,(8/n)^{0.82}$')
ax[2].set_xlabel('qubits $n$'); ax[2].set_ylabel(r'damping exponent $\xi$'); ax[2].legend(frameon=False); ax[2].set_ylim(0, 0.45); ax[2].set_xticks(xs_)
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_aer.pdf'); plt.close()
S_ = A_['summary']
for k, v in (('AerPearson', S_['pearson_r_lost']), ('AerMAEUncal', Cal['mae_test_uncalibrated']), ('AerMAECal', Cal['mae_test_calibrated']),
             ('AerBiasUncal', Cal['bias_test_uncalibrated']), ('AerBiasCal', Cal['bias_test_calibrated']), ('XiTrain', Cal['xi_train']),
             ('XiEight', Cal['xi_n8']), ('XiTen', Cal['xi_n10']), ('XiTwelve', Cal['xi_n12']), ('AerNPoints', S_['n_points'])):
    setnum(k, f'{v:.3f}' if isinstance(v, float) else v)


# ======================================================== extended Aer validation
X = json.load(open('results/aer_extended.json')); V, Th = X['vqe'], X['thermal']
for k, v in (('VqeN', V['n_points']), ('VqeXi', V['xi_train']), ('VqeMAEone', V['mae_test_xi1']), ('VqeMAEcal', V['mae_test_cal']),
             ('VqeCons', 100 * V['frac_xi1_conservative']), ('ThN', Th['n_points']), ('ThMAEesp', Th['mae_esp']), ('ThMAEcal', Th['mae_calibrated']),
             ('ThBiasCal', Th['bias_calibrated']), ('ThBiasEsp', Th['bias_esp']), ('ThOpt', 100 * Th['frac_cal_optimistic']), ('ThCons', 100 * Th['frac_esp_conservative'])):
    setnum(k, f'{v:.3f}' if isinstance(v, float) and k not in ('VqeCons', 'ThOpt', 'ThCons') else (f'{v:.0f}' if isinstance(v, float) else v))
with open(f'{OUT}/tables/tab_aerext.tex', 'w') as f:
    f.write('\\begin{tabular}{@{}lrrrr@{}}\\toprule\n')
    f.write('Check (Qiskit Aer, exact density matrix) & Points & Model & MAE & Bias\\\\\\midrule\n')
    f.write(f"VQE-type, depolarising, $n\\in\\{{4,6,8\\}}$ & {V['n_points']} & ESP ($\\xi{{=}}1$; used) & {V['mae_test_xi1']:.3f} & {V['bias_test_xi1']:+.3f}\\\\\n")
    f.write(f"\\quad (held-out Hamiltonian) & & fitted $\\xi{{=}}{V['xi_train']:.2f}$ & {V['mae_test_cal']:.3f} & --\\\\\\midrule\n")
    f.write(f"QAOA, thermal relaxation + depolarising floor, $n\\in\\{{8,10\\}}$ & {Th['n_points']} & ESP ($\\xi{{=}}1$) & {Th['mae_esp']:.3f} & {Th['bias_esp']:+.3f}\\\\\n")
    f.write(f"\\quad (no refitting) & & calibrated $\\xi(n)$ (used) & {Th['mae_calibrated']:.3f} & {Th['bias_calibrated']:+.3f}\\\\\n")
    f.write('\\bottomrule\\end{tabular}\n')

# ======================================================== decision latency
import policies, time as _t
e = E.Env(); t0 = _t.time(); nd = 0
for sd in range(3):
    e.reset(900 + sd, theta=0.5, lam_per_h=6.0)
    while not e.done:
        a = policies.planner(e); e.step(a); nd += 1
setnum('DecisionMs', f'{1000*(_t.time()-t0)/nd:.2f}')

# ======================================================== carbon trace figure
ZN = ['US-CAL-CISO', 'GB', 'DE', 'IN-WE', 'FR']
ZC = [C['orange'], C['blue'], C['grey'], C['verm'], C['green']]
R_ = pd.read_csv('data/traces_2025.csv')
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for z, col in zip(ZN, ZC):
    ax.plot(np.arange(168), R_[z].to_numpy()[4500:4500 + 168], color=col, lw=.8, label=z)
ax.set_xlabel('hour of week (sample week from 2025)'); ax.set_ylabel(r'$I(t)$ (gCO$_2$/kWh)'); ax.legend(frameon=False, ncol=3, fontsize=5.5)
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_traces.pdf'); plt.close()
for z in ZN:
    I = R_[z].to_numpy(); k = ''.join(ch for ch in z if ch.isalpha())
    setnum(f'Trace{k}', f'{I.mean():.0f}'); setnum(f'Trace{k}min', f'{np.percentile(I,5):.0f}'); setnum(f'Trace{k}max', f'{np.percentile(I,95):.0f}')
    setnum(f'Trace{k}acf', f'{np.corrcoef(I[:-24], I[24:])[0,1]:.2f}')


# ======================================================== IBM hardware (ibm_fez)
H_ = json.load(open('results/ibm_hardware.json'))
hm = np.array([x['lost_meas'] for x in H_]); he = np.array([x['lost_esp'] for x in H_]); hc = np.array([x['lost_cal'] for x in H_])
fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.75), sharey=True)
for a_, (pred, ttl) in zip(ax, ((he, r'ESP ($\xi=1$)'), (hc, r'Aer-calibrated $\xi$'))):
    for n_, col in ((6, C['blue']), (8, C['orange']), (10, C['green'])):
        ii = [i for i, x in enumerate(H_) if x['n'] == n_]; a_.scatter(hm[ii], pred[ii], s=9, color=col, label=f'n={n_}')
    a_.plot([0, .4], [0, .4], 'k--', lw=.6); a_.set_xlim(0, .4); a_.set_ylim(0, .4); a_.set_title(ttl, fontsize=6); a_.set_xlabel('measured loss (ibm_fez)')
ax[0].set_ylabel('predicted loss'); ax[0].legend(frameon=False, fontsize=5)
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_hw.pdf'); plt.close()
for k, v in (('HwN', len(H_)), ('HwMAEesp', np.abs(he - hm).mean()), ('HwBiasEsp', (he - hm).mean()), ('HwMAEcal', np.abs(hc - hm).mean()),
             ('HwBiasCal', (hc - hm).mean()), ('HwPearsonEsp', np.corrcoef(he, hm)[0, 1]), ('HwPearsonCal', np.corrcoef(hc, hm)[0, 1]),
             ('HwMeasMean', hm.mean()), ('HwMeasMax', hm.max()), ('HwMaxAbsEsp', np.abs(he - hm).max())):
    setnum(k, f'{v:.3f}' if k != 'HwN' else str(v))

# ======================================================== write macros
with open(f'{OUT}/numbers.tex', 'w') as f:
    for k, v in NUM.items():
        kk = ''.join(ch for ch in k if ch.isalpha())
        # digits -> letters so macro names are legal
        kk = k
        for d_, l_ in zip('0123456789', 'ZOTHFVSEGN'):
            kk = kk.replace(d_, l_)
        kk = ''.join(ch for ch in kk if ch.isalpha())
        f.write(f'\\newcommand{{\\n{kk}}}{{{v}}}\n')
json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in NUM.items()}, open('results/numbers.json', 'w'), indent=1, default=str)
print('macros', len(NUM))
