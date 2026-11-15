"""Baseline schedulers.  All use the same profiling/telemetry-derived predictions
(env.plans) as the learned agent, so the comparison is information-fair."""
import numpy as np
from env import K, N_ACT, act_decode


def _valid(env):
    return [a for a in range(N_ACT) if env.mask[a]]


def _j(env):
    return env.jobs[env.idx]


def classical_only(env):
    return 0


def random_feasible(env):
    return int(env.rng.choice(_valid(env)))


def fidelity_first(env):
    """Always offload to the lowest-noise QPU that has enough qubits, with mitigation on,
    no energy/queue/feasibility check (the 'fidelity-first' baseline of the outline)."""
    best, bF = 0, -1
    for k in range(K):
        a = 2 + (k * 2 + 1) * 2
        p = env.plans[a]
        if p is not None and p["F"] > bF:
            best, bF = a, p["F"]
    return best


def _now_actions(env):
    return [a for a in _valid(env) if act_decode(a)[2] == 0]


def speed_first(env):
    """Greedy static offloading: resource with earliest predicted completion among feasible."""
    return min(_now_actions(env), key=lambda a: env.plans[a]["end"])


def carbon_shift(env):
    """Speed-first routing + temporal shifting to the forecast-greenest window (Wiesner-style)."""
    a = speed_first(env)
    tgt, mit, _ = act_decode(a)
    b = (1 if tgt < 0 else a + 1)
    if env.mask[b] and env.plans[b]["end"] <= _j(env).deadline:
        return b
    return a


def energy_greedy(env):
    j = _j(env)
    acts = _now_actions(env)
    ok = [a for a in acts if env.plans[a]["end"] <= j.deadline]
    if ok:
        return min(ok, key=lambda a: env.plans[a]["E"])
    return min(acts, key=lambda a: env.plans[a]["end"])


def planner(env):
    """Myopic model-based planner: argmin of the one-step predicted cost (same cost as the RL reward),
    restricted to shield-feasible actions."""
    j, c = _j(env), env.cfg
    w = c["w"]
    best, bc = None, 1e18
    for a in _valid(env):
        p = env.plans[a]
        lat = p["end"] - j.arr
        cost = (w["alpha"] * p["E"] / c["E0"] + w["beta"] * p["C"] / c["C0"]
                + w["gamma"] * min(lat / j.slack, 3.0) + w["lam"] * float(p["end"] > j.deadline))
        if cost < bc:
            best, bc = a, cost
    return best


POLICIES = {
    "Classical-only": classical_only,
    "Random-feasible": random_feasible,
    "Fidelity-first": fidelity_first,
    "Speed-first": speed_first,
    "Carbon-shift": carbon_shift,
    "Energy-greedy": energy_greedy,
    "Planner": planner,
}


def run_episode(env, policy, seed, **over):
    obs, mask = env.reset(seed, **over)
    while not env.done:
        a = policy(env)
        (obs, mask), r, d, _ = env.step(a)
    return env.summary()


def make_planner(margin=0.0, eta=0.0):
    """EcoQ-MPC: shield-feasible one-step planner with (i) safety margin on predicted bias and
    (ii) congestion price eta * dur_h * min(wait_h, 10) approximating the externality of occupying a resource."""
    def pol(env):
        j, c = env.jobs[env.idx], env.cfg
        w = c["w"]
        best, bc = None, 1e18
        for a in range(N_ACT):
            if not env.mask[a]:
                continue
            p = env.plans[a]
            if p["tgt"] >= 0 and margin > 0 and p["bias"] > j.tol * (1 - margin):
                continue
            lat = p["end"] - j.arr
            wait_h = max(p["start"] - env.now, 0.0) / 3600.0
            cost = (w["alpha"] * p["E"] / c["E0"] + w["beta"] * p["C"] / c["C0"]
                    + w["gamma"] * min(lat / j.slack, 3.0) + w["lam"] * float(p["end"] > j.deadline)
                    + eta * (p["dur"] / 3600.0) * min(wait_h, 10.0))
            if cost < bc:
                best, bc = a, cost
        if best is None:
            best = int(np.flatnonzero(env.mask)[0])
        return best
    return pol
