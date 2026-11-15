"""Evaluation harness: main grid, ablations, sensitivity, Pareto.  Usage: python run_experiments.py <mode>"""
import sys, json, time, itertools, multiprocessing as mp
sys.path.insert(0, '/home/claude/ecoq')
import numpy as np, pandas as pd

THETAS = [0.0, 0.2, 0.35, 0.5]; LAMS = [2.0, 4.0, 6.0]; REGS = ['solar', 'wind', 'fossil']
W0 = dict(alpha=0.5, beta=0.5, gamma=0.3, lam=2.0, delta=1.0)
_cache = {}


def _agent(name):
    import torch, ppo
    if name not in _cache:
        d = torch.load(f'results/models/{name}.pt', weights_only=False)
        ag = ppo.Agent(d['obs_dim'])
        ag.pi.load_state_dict(d['pi']); ag.vf.load_state_dict(d['vf']); ag.norm.__dict__.update(d['norm'])
        _cache[name] = ag
    return _cache[name]


def make_policy(name):
    import policies, ppo
    if name in policies.POLICIES:
        return policies.POLICIES[name]
    if name in ('EcoQ', 'EcoQ-Oracle'):
        return policies.planner
    if name.startswith('PPO-'):
        kind, seed = name[4:].split('_s')[0], name.split('_s')[1]
        return ppo.agent_policy(_agent(f"{kind.lower()}_s{seed}"))
    raise KeyError(name)


def work(task):
    import env as E, policies
    name, th, lam, rg, seed, over, ablate = task
    ablate = set(ablate)
    over = dict(over)
    if name == 'EcoQ-Oracle':
        ablate |= {'oracle', 'perfect_fc'}
        over.update(sigma_pred=0.0, sigma_tel=0.0)
    e = E.Env(ablate=ablate)
    pol = make_policy(name)
    s = policies.run_episode(e, pol, seed, theta=th, lam_per_h=lam, regime=rg, **over)
    row = {k: v for k, v in s.items() if k not in ('util_each', 'share_k')}
    row.update(policy=name, theta=th, lam=lam, regime=rg, seed=seed,
               s0=s['share_k'][0], s1=s['share_k'][1], s2=s['share_k'][2])
    return row


def seed_of(ti, li, ri, ep, base=100000):
    return base + 10007 * ti + 1009 * li + 101 * ri + ep


def run(tasks, out, procs=2):
    t = time.time()
    with mp.Pool(procs) as p:
        rows = []
        for i, r in enumerate(p.imap(work, tasks, chunksize=4)):
            rows.append(r)
            if i % 200 == 0:
                print(f'{i}/{len(tasks)} {time.time()-t:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, len(rows), f'{time.time()-t:.0f}s')


def main_tasks(policies_, n_ep):
    T = []
    for (ti, th), (li, lam), (ri, rg) in itertools.product(enumerate(THETAS), enumerate(LAMS), enumerate(REGS)):
        for ep in range(n_ep):
            sd = seed_of(ti, li, ri, ep)
            for p in policies_:
                T.append((p, th, lam, rg, sd, {}, []))
    return T


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'main_heur':
        P = ['Classical-only', 'Random-feasible', 'Fidelity-first', 'Speed-first', 'Carbon-shift', 'Energy-greedy', 'EcoQ', 'EcoQ-Oracle']
        run(main_tasks(P, 6), 'results/main_heur.csv')
    elif mode == 'main_rl':
        P = ['PPO-BC_s0', 'PPO-BC_s1', 'PPO-BC_s2', 'PPO-SCRATCH_s0', 'PPO-SCRATCH_s1']
        run(main_tasks(P, 6), 'results/main_rl.csv')
    elif mode == 'ablate':
        V = {
            'EcoQ (full)': ({}, []),
            'w/o carbon term': (dict(w={**W0, 'beta': 0.0}), []),
            'w/o energy term': (dict(w={**W0, 'alpha': 0.0}), []),
            'w/o temporal shifting': ({}, ['no_defer']),
            'w/o mitigation choice': ({}, ['no_mit']),
            'w/o fidelity shield': ({}, ['no_shield']),
            'w/o set-up awareness': (dict(setup_aware=False), []),
            'w/o SLO/latency terms': (dict(w={**W0, 'gamma': 0.0, 'lam': 0.0}), []),
        }
        T = []
        for (ti, th), (li, lam), (ri, rg) in itertools.product(enumerate(THETAS), enumerate(LAMS), enumerate(REGS)):
            if th < 0.3:
                continue
            for ep in range(4):
                sd = seed_of(ti, li, ri, ep, base=300000)
                for k, (ov, ab) in V.items():
                    T.append(('EcoQ', th, lam, rg, sd, ov, ab))
        # label variants via a side table
        names = [k for k in V for _ in range(1)]
        rows_T = T
        t = time.time()
        # run with variant label carried in 'over' key
        lab = {id(t_): None for t_ in T}
        tasks = []
        for t_ in T:
            nm = [k for k, (ov, ab) in V.items() if ov == t_[5] and ab == t_[6]][0]
            tasks.append(t_)
        with mp.Pool(2) as p:
            out = p.map(work, tasks, chunksize=4)
        # recover labels by order
        labels = []
        for (ti, th), (li, lam), (ri, rg) in itertools.product(enumerate(THETAS), enumerate(LAMS), enumerate(REGS)):
            if th < 0.3:
                continue
            for ep in range(4):
                labels += list(V.keys())
        df = pd.DataFrame(out); df['variant'] = labels
        df.to_csv('results/ablation.csv', index=False); print('saved ablation', len(df), time.time() - t)
        # oracle variant
        P = [('EcoQ-Oracle', th, lam, rg, seed_of(ti, li, ri, ep, base=300000), {}, [])
             for (ti, th), (li, lam), (ri, rg) in itertools.product(enumerate(THETAS), enumerate(LAMS), enumerate(REGS))
             if th >= 0.3 for ep in range(4)]
        run(P, 'results/ablation_oracle.csv')
    elif mode == 'sens':
        S = {
            'nominal': {},
            'xi=1 (uncalibrated ESP)': dict(xi_mode='esp'),
            'telemetry noise x4': dict(sigma_tel=0.20),
            'forecast error x3': dict(f_s0=0.09, f_s1=0.012),
            'perfect forecast': dict(f_s0=0.0, f_s1=0.0),
            'profiling error 0.6': dict(sigma_pred=0.6),
            'profiling error 0.1': dict(sigma_pred=0.1),
            'set-up 0 s': dict(tau_setup=0.0),
            'set-up 1800 s': dict(tau_setup=1800.0, tau_idle=3600.0),
            'mitigation rho=0.3': dict(rho_mit=0.3),
            'drift eta=1.0': dict(drift_eta=1.0),
            'calibration noise sigma_k=0.5': dict(sigma_kappa=0.5),
        }
        T, lab = [], []
        for k, ov in S.items():
            for ri, rg in enumerate(REGS):
                for ep in range(4):
                    sd = seed_of(3, 1, ri, ep, base=500000)
                    for p in ['Classical-only', 'Speed-first', 'EcoQ']:
                        T.append((p, 0.5, 4.0, rg, sd, ov, [])); lab.append(k)
        with mp.Pool(2) as pool:
            out = pool.map(work, T, chunksize=4)
        df = pd.DataFrame(out); df['setting'] = lab
        df.to_csv('results/sensitivity.csv', index=False); print('saved sens', len(df))
    elif mode == 'pareto':
        Wset = {'E-only': dict(alpha=1.0, beta=0.0), 'E-heavy': dict(alpha=0.75, beta=0.25), 'balanced': dict(alpha=0.5, beta=0.5),
                'C-heavy': dict(alpha=0.25, beta=0.75), 'C-only': dict(alpha=0.0, beta=1.0),
                'C-only, low latency weight': dict(alpha=0.0, beta=1.0, gamma=0.05, lam=1.0),
                'balanced, high latency weight': dict(alpha=0.5, beta=0.5, gamma=1.5, lam=4.0)}
        T, lab = [], []
        for k, wv in Wset.items():
            for ri, rg in enumerate(REGS):
                for ep in range(6):
                    sd = seed_of(3, 1, ri, ep, base=700000)
                    T.append(('EcoQ', 0.5, 4.0, rg, sd, dict(w={**W0, **wv}), [])); lab.append(k)
        for p in ['Classical-only', 'Speed-first', 'Carbon-shift', 'Energy-greedy']:
            for ri, rg in enumerate(REGS):
                for ep in range(6):
                    sd = seed_of(3, 1, ri, ep, base=700000)
                    T.append((p, 0.5, 4.0, rg, sd, {}, [])); lab.append(p)
        with mp.Pool(2) as pool:
            out = pool.map(work, T, chunksize=4)
        df = pd.DataFrame(out); df['setting'] = lab
        df.to_csv('results/pareto.csv', index=False); print('saved pareto', len(df))
