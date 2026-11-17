"""Masked PPO (Schulman et al. 2017) for the EcoQ scheduling MDP."""
import time, copy, json, numpy as np, torch, torch.nn as nn
from env import Env, N_ACT, DEFAULTS

torch.set_num_threads(1)
THETAS = [0.0, 0.2, 0.35, 0.5]
LAMS = [2.0, 4.0, 6.0]
REGIMES = ["US-CAL-CISO", "GB", "DE", "IN-WE", "FR"]


def mlp(i, o, h=128, last_gain=1.0):
    l = [nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o)]
    for m in l:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 2 ** 0.5); nn.init.zeros_(m.bias)
    nn.init.orthogonal_(l[-1].weight, last_gain)
    return nn.Sequential(*l)


class RunNorm:
    def __init__(self, d):
        self.n, self.m, self.v = 1e-4, np.zeros(d), np.ones(d)

    def update(self, x):
        x = np.asarray(x)
        bm, bv, bn = x.mean(0), x.var(0), len(x)
        d = bm - self.m; tot = self.n + bn
        self.m = self.m + d * bn / tot
        self.v = (self.v * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.m) / np.sqrt(self.v + 1e-8), -10, 10)


class Agent:
    def __init__(self, obs_dim, seed=0):
        torch.manual_seed(seed)
        self.pi, self.vf = mlp(obs_dim, N_ACT, last_gain=0.01), mlp(obs_dim, 1, last_gain=1.0)
        self.norm = RunNorm(obs_dim)

    def logits(self, x, mask):
        lg = self.pi(x)
        return lg.masked_fill(~mask, -1e9)

    def act(self, obs, mask, greedy=False):
        x = torch.as_tensor(self.norm(obs), dtype=torch.float32)[None]
        m = torch.as_tensor(mask)[None]
        with torch.no_grad():
            lg = self.logits(x, m)
            if greedy:
                return int(lg.argmax(-1))
            d = torch.distributions.Categorical(logits=lg)
            a = d.sample()
            return int(a), float(d.log_prob(a)), float(self.vf(x))


def sample_scenario(rng):
    return dict(theta=float(rng.choice(THETAS)), lam_per_h=float(rng.choice(LAMS)),
                regime=str(rng.choice(REGIMES)), trace_split="train")


def train(seed=0, steps=150_000, ablate=None, w=None, cfg=None, log_every=1, verbose=False, init_agent=None,
          init='bc', lr0=5e-5, val_every=4, bc_steps=40_000, rbase=True, kl_coef=0.3, NS=4096):
    rng = np.random.default_rng(10_000 + seed)
    cfgd = dict(cfg or {})
    if w:
        cfgd["w"] = {**DEFAULTS["w"], **w}
    env = Env(cfgd, ablate=ablate)
    obs, mask = env.reset(int(rng.integers(1e9)), **sample_scenario(rng))
    ag = init_agent or Agent(len(obs), seed)
    bc_acc = None; ref = None
    if init == 'bc':
        bc_acc = bc_pretrain(ag, cfgd, ablate, n_steps=bc_steps, seed=seed, rbase=rbase)
        ref = copy.deepcopy(ag.pi)
    else:
        lr0 = 3e-4
    best = dict(score=1e9, state=None, step=0)
    base_costs = None
    sc0, base_costs = validate(ag, cfgd, ablate)
    best.update(score=sc0, state=(copy.deepcopy(ag.pi.state_dict()), copy.deepcopy(ag.vf.state_dict()), copy.deepcopy(ag.norm.__dict__)), step=0)
    if verbose:
        print('init', init, 'bc_acc', bc_acc, 'val ratio', round(sc0, 4), flush=True)
    opt = torch.optim.Adam(list(ag.pi.parameters()) + list(ag.vf.parameters()), lr=3e-4, eps=1e-5)
    EP, MB, GAM, LAM, CLIP, RS = 6, 512, 0.95, 0.95, 0.15, 5.0
    ep_ret, ep_rets, curve = 0.0, [], []
    t0, done_steps = time.time(), 0
    n_up = steps // NS
    for up in range(n_up):
        frac = 1 - up / n_up
        for g in opt.param_groups:
            g["lr"] = lr0 * (0.1 + 0.9 * frac)
        ent_c = (0.003 if init == 'bc' else 0.01) * (0.1 + 0.9 * frac)
        O, M, A, LP, R, V, D = [], [], [], [], [], [], []
        for _ in range(NS):
            O.append(obs); M.append(mask)
            ag.norm.update(obs[None]) if len(O) % 16 == 0 else None
            a, lp, v = ag.act(obs, mask)
            (obs2, mask2), r, d, info = env.step(a)
            if rbase:
                r = -(info['cost'] - info['base'])
            A.append(a); LP.append(lp); R.append(r / RS); V.append(v); D.append(d)
            ep_ret += r
            if d:
                ep_rets.append(ep_ret); ep_ret = 0.0
                obs2, mask2 = env.reset(int(rng.integers(1e9)), **sample_scenario(rng))
            obs, mask = obs2, mask2
        ag.norm.update(np.array(O))
        x_last = torch.as_tensor(ag.norm(obs), dtype=torch.float32)[None]
        with torch.no_grad():
            last_v = float(ag.vf(x_last))
        adv = np.zeros(NS, dtype=np.float32); lastgae = 0.0
        for t in reversed(range(NS)):
            nv = last_v if t == NS - 1 else V[t + 1]
            nd = 1.0 - float(D[t])
            delta = R[t] + GAM * nv * nd - V[t]
            lastgae = delta + GAM * LAM * nd * lastgae
            adv[t] = lastgae
        ret = adv + np.array(V, dtype=np.float32)
        X = torch.as_tensor(ag.norm(np.array(O)), dtype=torch.float32)
        Mk = torch.as_tensor(np.array(M)); At = torch.as_tensor(np.array(A)); LPo = torch.as_tensor(np.array(LP), dtype=torch.float32)
        Adv = torch.as_tensor(adv); Ret = torch.as_tensor(ret)
        for _ in range(EP):
            perm = torch.randperm(NS)
            for i in range(0, NS, MB):
                ix = perm[i:i + MB]
                dist = torch.distributions.Categorical(logits=ag.logits(X[ix], Mk[ix]))
                lp = dist.log_prob(At[ix])
                ratio = torch.exp(lp - LPo[ix])
                a_ = (Adv[ix] - Adv[ix].mean()) / (Adv[ix].std() + 1e-8)
                l_pi = -torch.min(ratio * a_, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_).mean()
                l_v = 0.5 * ((ag.vf(X[ix]).squeeze(-1) - Ret[ix]) ** 2).mean()
                loss = l_pi + 0.5 * l_v - ent_c * dist.entropy().mean()
                if ref is not None and kl_coef > 0:
                    with torch.no_grad():
                        lp0 = torch.log_softmax(ref(X[ix]).masked_fill(~Mk[ix], -1e9), -1)
                    lpn = torch.log_softmax(ag.logits(X[ix], Mk[ix]), -1)
                    kl = (lpn.exp() * (lpn - lp0)).sum(-1).mean()
                    loss = loss + kl_coef * kl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(ag.pi.parameters()) + list(ag.vf.parameters()), 0.5)
                opt.step()
        recent = float(np.mean(ep_rets[-10:])) if ep_rets else float("nan")
        rec = dict(update=up, steps=(up + 1) * NS, ep_return=recent, time=time.time() - t0)
        if (up + 1) % val_every == 0 or up == n_up - 1:
            sc, _ = validate(ag, cfgd, ablate, base_costs)
            rec["val_ratio"] = sc
            if sc < best["score"]:
                best.update(score=sc, step=(up + 1) * NS, state=(copy.deepcopy(ag.pi.state_dict()), copy.deepcopy(ag.vf.state_dict()), copy.deepcopy(ag.norm.__dict__)))
            if verbose:
                print(f"  val ratio {sc:.4f} (best {best['score']:.4f} @ {best['step']})", flush=True)
        curve.append(rec)
        if verbose and up % 5 == 0:
            print(f"up {up}/{n_up} steps {(up+1)*NS} ret {recent:.1f} t {time.time()-t0:.0f}s", flush=True)
    ag.pi.load_state_dict(best["state"][0]); ag.vf.load_state_dict(best["state"][1]); ag.norm.__dict__.update(best["state"][2])
    ag.best_step, ag.best_val, ag.bc_acc = best["step"], best["score"], bc_acc
    return ag, curve


def collect_planner(env_cfg, ablate, n_steps, seed, rbase=True):
    """Roll out the myopic planner under randomised scenarios -> (obs, mask, action, discounted return)."""
    from policies import planner
    rng = np.random.default_rng(777_000 + seed)
    env = Env(env_cfg, ablate=ablate)
    O, M, A, R, D = [], [], [], [], []
    while len(O) < n_steps:
        obs, mask = env.reset(int(rng.integers(1e9)), **sample_scenario(rng))
        while not env.done:
            a = planner(env)
            O.append(obs); M.append(mask); A.append(a)
            (obs, mask), r, d, info = env.step(a)
            R.append((-(info['cost'] - info['base']) if rbase else r) / 5.0); D.append(d)
    ret = np.zeros(len(R), dtype=np.float32); g = 0.0
    for t in reversed(range(len(R))):
        g = R[t] + 0.95 * g * (0.0 if D[t] else 1.0) if not D[t] else R[t]
        ret[t] = g
    return np.array(O), np.array(M), np.array(A), ret


def bc_pretrain(ag, env_cfg, ablate, n_steps=40_000, epochs=30, seed=0, rbase=True):
    O, M, A, R = collect_planner(env_cfg, ablate, n_steps, seed, rbase)
    ag.norm.update(O)
    X = torch.as_tensor(ag.norm(O), dtype=torch.float32); Mk = torch.as_tensor(M)
    At = torch.as_tensor(A); Rt = torch.as_tensor(R)
    opt = torch.optim.Adam(list(ag.pi.parameters()) + list(ag.vf.parameters()), lr=1e-3)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            ix = perm[i:i + 512]
            lg = ag.logits(X[ix], Mk[ix])
            l_pi = nn.functional.cross_entropy(lg, At[ix])
            l_v = 0.5 * ((ag.vf(X[ix]).squeeze(-1) - Rt[ix]) ** 2).mean()
            opt.zero_grad(); (l_pi + l_v).backward(); opt.step()
    with torch.no_grad():
        acc = float((ag.logits(X, Mk).argmax(-1) == At).float().mean())
    return acc


VAL = [dict(theta=th, lam_per_h=lam, regime=rg, trace_split="train") for th, lam, rg in
       [(0.35, 2.0, "US-CAL-CISO"), (0.5, 2.0, "GB"), (0.5, 4.0, "DE"), (0.5, 6.0, "IN-WE"), (0.35, 6.0, "FR"),
        (0.2, 4.0, "US-CAL-CISO"), (0.5, 6.0, "GB"), (0.35, 4.0, "DE")]]


def validate(ag, env_cfg, ablate, base_costs=None):
    """Mean of cost(agent)/cost(planner) over fixed validation episodes (lower = better)."""
    from policies import planner, run_episode
    env = Env(env_cfg, ablate=ablate)
    pol = agent_policy(ag)
    costs = []
    for i, sc in enumerate(VAL):
        costs.append(run_episode(env, pol, 31_000 + i, **sc)["cost"])
    if base_costs is None:
        base_costs = [run_episode(env, planner, 31_000 + i, **sc)["cost"] for i, sc in enumerate(VAL)]
    return float(np.mean([c / max(b, 1e-6) for c, b in zip(costs, base_costs)])), base_costs


def agent_policy(ag):
    def pol(env):
        obs, _ = env._obs()
        return ag.act(obs, env.mask, greedy=True)
    return pol
