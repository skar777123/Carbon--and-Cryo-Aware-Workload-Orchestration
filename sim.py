"""EcoQ-Orchestrator: discrete-event simulator of a hybrid quantum-classical data centre.

Every physical parameter in `Cfg` is a MODELLING ASSUMPTION (see paper, Table of assumptions);
none is calibrated against a specific production device. Time unit: hours.
"""
import math
import heapq
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Cfg:
    H: float = 336.0                 # horizon (14 days)
    pue: float = 1.25                # facility overhead on classical + host + control-electronics increments
    # ---- classical pool
    n_nodes: int = 32
    p_node_kw: float = 3.0
    t_cl0_s: float = 15.0            # classical time-to-target at n_ref qubits
    n_ref: int = 16
    kappa: float = 2.5               # classical time doubles every kappa qubits (advantage regime parameter)
    t_cl_max_h: float = 6.0          # classical solver wall-clock cap -> larger jobs are classically infeasible
    # ---- QPU fleet (superconducting, dilution-refrigerated)
    K: int = 3
    e2q: tuple = (0.004, 0.006, 0.009)
    ero: tuple = (0.012, 0.020, 0.030)
    drift: float = 0.015             # relative 2q-error growth per hour since calibration
    cal_period: float = 24.0
    cal_dur: float = 0.75
    cal_off: tuple = (3.0, 11.0, 19.0)
    esp_min: float = 0.08
    p_fridge_kw: float = 24.0        # compressor + chilled-water heat rejection while cold
    p_idle_kw: float = 2.0           # control electronics, idle (while cold)
    p_run_kw: float = 3.0            # control electronics, executing
    p_host_kw: float = 1.0           # host/optimiser draw during classical half of variational loop
    t_shot_s: float = 200e-6
    t_loop_s: float = 0.25           # classical optimiser + interconnect latency per variational iteration
    t_setup_s: float = 120.0         # session / compile warm-up when QPU idle > tau
    tau_h: float = 0.25
    t_cool_h: float = 30.0           # cooldown duration (assumption)
    cool_mult: float = 1.5           # cooldown draw relative to p_fridge
    # ---- workload
    lam0: float = 8.0                # mean arrivals / hour
    wknd: float = 0.35
    # ---- cost / control
    mu: float = 0.10                 # kgCO2e-equivalent price of one kWh (energy weight)
    zeta: float = 10.0               # kgCO2e-equivalent penalty (Lagrange multiplier) per SLA miss / unserved job
    lam_wait: float = 0.05           # kgCO2e-equivalent cost per hour of waiting (latency term)
    max_defer: int = 12              # max deferrals per job
    s_c: float = 0.5                 # share of cryo hold cost charged to QPU busy time in the training cost
    rho_star: float = 0.6            # target utilisation of load-following fleet controller
    win_h: float = 72.0


# ------------------------------------------------------------------ traces
def _ar1(rng, T, phi, sig):
    x = np.zeros(T)
    x[0] = rng.normal(0, sig / math.sqrt(1 - phi ** 2))
    e = rng.normal(0, sig, T)
    for i in range(1, T):
        x[i] = phi * x[i - 1] + e[i]
    return x


def gen_ci(profile, rng, T):
    """Synthetic hourly marginal-carbon-intensity trace (gCO2e/kWh). Stylised, NOT real grid data."""
    h = np.arange(T)
    hod = h % 24
    wk = (((h // 24) % 7) >= 5).astype(float)
    if profile == "solar":
        sun = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) ** 1.5
        ci = 260 - 150 * sun + 70 * np.exp(-0.5 * ((hod - 19.5) / 1.5) ** 2) - 25 * wk + _ar1(rng, T, 0.85, 18)
    elif profile == "wind":
        ci = 220 + 25 * np.cos(2 * np.pi * (hod - 19) / 24) + _ar1(rng, T, 0.985, 9)
    elif profile == "fossil":
        ci = 620 + 30 * np.cos(2 * np.pi * (hod - 18) / 24) + _ar1(rng, T, 0.9, 12)
    else:
        raise ValueError(profile)
    return np.clip(ci, 30, 900)


PROFILES = ("solar", "wind", "fossil")


class Job:
    __slots__ = ("arr", "n", "kind", "layers", "iters", "n0", "slack", "deadline", "n2q", "n1q", "cls", "jid", "ndef")


def gen_jobs(cfg, rng):
    hod = np.arange(24)
    shape = 0.45 + 1.1 * np.exp(-0.5 * ((hod - 14) / 4.5) ** 2)
    shape = shape / shape.mean()
    wf = (5 + 2 * cfg.wknd) / 7.0
    Hh = int(cfg.H)
    lam = np.array([cfg.lam0 * shape[h % 24] * (cfg.wknd if (h // 24) % 7 >= 5 else 1.0) / wf for h in range(Hh)])
    lmax = lam.max()
    N = rng.poisson(lmax * cfg.H)
    ts = np.sort(rng.uniform(0, cfg.H, N))
    keep = rng.uniform(0, lmax, N) < lam[np.minimum(ts.astype(int), Hh - 1)]
    jobs = []
    for i, t in enumerate(ts[keep]):
        j = Job()
        j.jid = i
        j.ndef = 0
        j.arr = float(t)
        u = rng.random()
        if u < 0.45:
            j.cls, j.n, j.layers = 0, int(rng.integers(12, 27)), int(rng.integers(2, 6))
        elif u < 0.83:
            j.cls, j.n, j.layers = 1, int(rng.integers(27, 41)), int(rng.integers(2, 6))
        else:
            j.cls, j.n, j.layers = 2, int(rng.integers(43, 47)), int(rng.integers(1, 3))
        j.kind = int(rng.random() < 0.5)  # 0 = QAOA(MaxCut, 3-regular), 1 = VQE (hardware-efficient)
        if j.kind == 0:
            j.n2q, j.n1q = 3 * j.n * j.layers, j.n * (j.layers + 1)
        else:
            j.n2q, j.n1q = (j.n - 1) * j.layers, 2 * j.n * j.layers
        j.iters = int(rng.integers(80, 301))
        j.n0 = int(rng.choice([2000, 5000]))
        v = rng.random()
        j.slack = 1.0 if v < 0.3 else (6.0 if v < 0.7 else 24.0)
        if j.cls == 2:
            j.slack = max(j.slack, 6.0)
        # every job must be meetable in principle: slack >= 1.5 x best-case (unloaded, freshly calibrated) runtime
        tmin = 1e9
        tcl = cfg.t_cl0_s * 2.0 ** ((j.n - cfg.n_ref) / cfg.kappa) / 3600.0
        if tcl <= cfg.t_cl_max_h:
            tmin = tcl
        e2 = cfg.e2q[0]
        esp0 = math.exp(j.n1q * math.log1p(-e2 / 10) + j.n2q * math.log1p(-e2) + j.n * math.log1p(-cfg.ero[0]))
        if esp0 >= cfg.esp_min:
            tq = (j.iters * j.n0 / (esp0 * esp0) * cfg.t_shot_s + j.iters * cfg.t_loop_s + cfg.t_setup_s) / 3600.0
            tmin = min(tmin, tq)
        if tmin < 1e8:
            j.slack = max(j.slack, 1.5 * tmin)
        j.deadline = j.arr + j.slack
        jobs.append(j)
    return jobs


class Ctx:
    __slots__ = ("job", "t", "plans", "mask", "env", "A")


class Episode:
    """One simulated horizon. prov in {'lf','all_on','classical'}."""

    def __init__(self, cfg, profile, seed, prov="lf"):
        self.cfg, self.profile, self.seed, self.prov = cfg, profile, seed, prov
        T = int(cfg.H) + 80
        self.T = T
        self.ci = gen_ci(profile, np.random.default_rng([seed, 1]), T)
        eps = _ar1(np.random.default_rng([seed, 3]), T, 0.9, 0.035)
        self.fc = np.clip(self.ci * (1 + eps), 20, 1000)
        self.cum_t = np.concatenate([[0], np.cumsum(self.ci)])
        self.cum_f = np.concatenate([[0], np.cumsum(self.fc)])
        self.jobs = gen_jobs(cfg, np.random.default_rng([seed, 2]))
        K = cfg.K
        self.A = K + 2  # 0 classical, 1..K QPU, K+1 defer
        if prov == "classical":
            self.state = ["warm"] * K
        elif prov.startswith("static"):
            nn = int(prov[6:])
            self.state = ["cold" if k < nn else "warm" for k in range(K)]
        else:
            self.state = ["cold"] * K
        self.arr_t = np.array([j.arr for j in self.jobs])
        self.arr_c = np.array([j.cls for j in self.jobs])
        self.prov_rewards, self.prov_steps, self.cc_prev = [], 0, 0.0
        self.cold_at = [0.0] * K
        self.free_at = [0.0] * K
        self.last_end = [-1e9] * K
        self.nodes = [0.0] * cfg.n_nodes
        self.busy_hr = np.zeros(T + 4)
        self.low_days = 0
        self.last_t = 0.0
        self.fleet_kwh = 0.0
        self.fleet_kg = 0.0
        self.cold_unit_h = 0.0
        self.m = dict(op_kwh=0.0, op_kg=0.0, n_cl=0, n_q=0, n_def=0, miss=0, unserved=0, wait=0.0, jobs=0,
                      qbusy=0.0, qkg_cost=0.0, per_unit=[0] * K, cooldowns=0, gates=0, decisions=0)
        self.rewards = []

    # ---------------- carbon integrals
    def _F(self, cum, arr, t):
        i = int(t)
        if i >= self.T:
            return cum[self.T]
        return cum[i] + arr[i] * (t - i)

    def ci_int(self, a, b, est=False):
        cum, arr = (self.cum_f, self.fc) if est else (self.cum_t, self.ci)
        return self._F(cum, arr, b) - self._F(cum, arr, a)

    # ---------------- device model
    def hsc(self, k, s):
        c = self.cfg
        return max(0.0, ((s - c.cal_off[k]) % c.cal_period) - c.cal_dur)

    def esp(self, job, k, hs):
        c = self.cfg
        e2 = min(0.2, c.e2q[k] * (1 + c.drift * hs))
        e1 = e2 / 10.0
        return math.exp(job.n1q * math.log1p(-e1) + job.n2q * math.log1p(-e2) + job.n * math.log1p(-c.ero[k]))

    def cal_shift(self, k, s, d):
        c = self.cfg
        off, P = c.cal_off[k], c.cal_period
        m = math.floor((s - off) / P)
        w0 = off + m * P
        w1 = w0 + c.cal_dur
        if s < w1:
            s = w1
        w0n = off + (m + 1) * P
        if s + d > w0n:
            s = w0n + c.cal_dur
        return s

    def next_cal_end(self, k, s):
        c = self.cfg
        off, P = c.cal_off[k], c.cal_period
        m = math.floor((s - off) / P)
        w1 = off + m * P + c.cal_dur
        return w1 if s < w1 else off + (m + 1) * P + c.cal_dur

    def plan_q(self, job, k, t, est=True):
        c = self.cfg
        st = self.state[k]
        if st == "warm":
            return None
        s = max(t, self.free_at[k], self.cold_at[k] if st == "cooling" else 0.0)
        for _ in range(3):
            esp = self.esp(job, k, self.hsc(k, s))
            if esp < c.esp_min:
                s = self.next_cal_end(k, s)
                continue
            setup = (s - self.last_end[k]) > c.tau_h
            shots_h = job.iters * job.n0 / (esp * esp) * c.t_shot_s / 3600.0
            loop_h = job.iters * c.t_loop_s / 3600.0
            setup_h = c.t_setup_s / 3600.0 if setup else 0.0
            d = shots_h + loop_h + setup_h
            s2 = self.cal_shift(k, s, d)
            if s2 > s + 1e-12:
                s = s2
                continue
            break
        else:
            return None
        e_it = (c.p_run_kw - c.p_idle_kw) * (shots_h + setup_h) + c.p_host_kw * loop_h
        E = c.pue * e_it
        ci = self.ci_int(s, s + d, est)
        kg = (E / d) * ci / 1000.0
        pc = c.p_fridge_kw + c.p_idle_kw
        sh_E = pc * d
        sh_kg = pc * ci / 1000.0
        cost = kg + c.mu * E + c.s_c * (sh_kg + c.mu * sh_E) + c.lam_wait * (s - t)
        return (s, s + d, esp, E, kg, cost, sh_E, sh_kg)

    def plan_c(self, job, t, est=True):
        c = self.cfg
        tcl = c.t_cl0_s * 2.0 ** ((job.n - c.n_ref) / c.kappa) / 3600.0
        if tcl > c.t_cl_max_h:
            return None
        s = max(t, self.nodes[0])
        E = c.p_node_kw * tcl * c.pue
        kg = c.p_node_kw * c.pue * self.ci_int(s, s + tcl, est) / 1000.0
        return (s, s + tcl, 1.0, E, kg, kg + c.mu * E + c.lam_wait * (s - t), 0.0, 0.0)

    def plan_all(self, job, t):
        return [self.plan_c(job, t)] + [self.plan_q(job, k, t) for k in range(self.cfg.K)]

    # ---------------- fleet accounting & provisioning
    def _fleet_power(self):
        c = self.cfg
        P = 0.0
        for k in range(c.K):
            if self.state[k] == "cold":
                P += c.p_fridge_kw + c.p_idle_kw
            elif self.state[k] == "cooling":
                P += c.cool_mult * c.p_fridge_kw
        return P

    def advance_fleet(self, t):
        te = min(t, self.cfg.H)
        while self.last_t < te - 1e-12:
            nxt = min([self.cold_at[k] for k in range(self.cfg.K) if self.state[k] == "cooling"], default=1e18)
            se = min(te, nxt)
            P = self._fleet_power()
            dt = se - self.last_t
            self.fleet_kwh += P * dt
            self.fleet_kg += P * self.ci_int(self.last_t, se) / 1000.0
            self.cold_unit_h += dt * sum(1 for s_ in self.state if s_ == "cold")
            self.last_t = se
            if se >= nxt - 1e-12:
                for k in range(self.cfg.K):
                    if self.state[k] == "cooling" and self.cold_at[k] <= se + 1e-12:
                        self.state[k] = "cold"
                        self.cold_at[k] = 0.0

    def _add_busy(self, s, e):
        h = int(s)
        while h < e and h < len(self.busy_hr):
            self.busy_hr[h] += min(e, h + 1.0) - max(s, float(h))
            h += 1

    def trailing_util(self, t):
        """(busy+backlog, busy) QPU-equivalents over the trailing window, from the hourly busy histogram."""
        i1 = int(t)
        i0 = max(0, int(t - self.cfg.win_h))
        W = i1 - i0
        if W <= 0:
            return 0.0, 0.0
        busy = float(self.busy_hr[i0:i1].sum())
        backlog = sum(max(0.0, self.free_at[k] - t) for k in range(self.cfg.K) if self.state[k] != "warm")
        return (busy + backlog) / W, busy / W

    def provision(self, t):
        if self.prov != "lf":
            return
        c = self.cfg
        if t <= 0:
            return
        u, _ = self.trailing_util(t)
        n_target = max(1, math.ceil(u / c.rho_star))
        committed = [k for k in range(c.K) if self.state[k] != "warm"]
        if n_target < len(committed):
            self.low_days += 1
            if self.low_days >= 2:
                for k in reversed(committed):
                    if self.state[k] == "cold" and self.free_at[k] <= t and len(committed) > 1:
                        self.state[k] = "warm"
                        self.m["gates"] += 1
                        break
        else:
            self.low_days = 0
        if n_target > len(committed):
            for k in range(c.K):
                if self.state[k] == "warm":
                    self.state[k] = "cooling"
                    self.cold_at[k] = t + c.t_cool_h
                    self.last_end[k] = -1e9
                    self.m["cooldowns"] += 1
                    break

    # ---------------- learned provisioning hooks (slow timescale)
    def cum_cost(self):
        c, m = self.cfg, self.m
        return (m["op_kg"] + self.fleet_kg + c.mu * (m["op_kwh"] + self.fleet_kwh)
                + c.zeta * (m["miss"] + m["unserved"]) + c.lam_wait * m["wait"])

    def prov_features(self, t):
        c = self.cfg
        i0 = min(int(t), self.T - 80)
        hw = (t % 168.0) / 168.0
        n_cold = sum(1 for s in self.state if s == "cold")
        n_cool = sum(1 for s in self.state if s == "cooling")
        n_warm = c.K - n_cold - n_cool
        cr = max([self.cold_at[k] - t for k in range(c.K) if self.state[k] == "cooling"], default=0.0)
        u72, _ = self.trailing_util(t)
        i1 = int(t)
        b24 = float(self.busy_hr[max(0, i1 - 24):i1].sum()) / max(min(24, i1), 1)
        backlog = sum(max(0.0, self.free_at[k] - t) for k in range(c.K) if self.state[k] != "warm")
        r24 = np.searchsorted(self.arr_t, t) - np.searchsorted(self.arr_t, t - 24.0)
        i72 = np.searchsorted(self.arr_t, t - 72.0)
        i_t = np.searchsorted(self.arr_t, t)
        nl72 = int((self.arr_c[i72:i_t] == 2).sum())
        past = self.ci[max(0, i0 - 24):i0 + 1].mean() if i0 > 0 else self.ci[0]
        x = np.array([math.sin(2 * math.pi * hw), math.cos(2 * math.pi * hw), n_cold / c.K, n_cool / c.K,
                      n_warm / c.K, cr / c.t_cool_h, min(u72, 3.0) / 3.0, min(b24, 3.0) / 3.0,
                      min(backlog, 48.0) / 24.0, r24 / 240.0, nl72 / 100.0, past / 500.0,
                      self.fc[i0:i0 + 24].mean() / 500.0, self.fc[i0:i0 + 72].mean() / 500.0,
                      (c.H - t) / c.H])
        return x

    def prov_mask(self, t):
        c = self.cfg
        committed = [k for k in range(c.K) if self.state[k] != "warm"]
        can_gate = len(committed) > 1 and any(self.state[k] == "cold" and self.free_at[k] <= t for k in committed)
        can_cool = len(committed) < c.K
        return np.array([True, can_gate, can_cool])

    def apply_prov(self, a, t):
        c = self.cfg
        if a == 1:
            committed = [k for k in range(c.K) if self.state[k] != "warm"]
            if len(committed) > 1:
                for k in reversed(committed):
                    if self.state[k] == "cold" and self.free_at[k] <= t:
                        self.state[k] = "warm"
                        self.m["gates"] += 1
                        return
        elif a == 2:
            for k in range(c.K):
                if self.state[k] == "warm":
                    self.state[k] = "cooling"
                    self.cold_at[k] = t + c.t_cool_h
                    self.last_end[k] = -1e9
                    self.m["cooldowns"] += 1
                    return

    # ---------------- decision context
    def make_ctx(self, job, t):
        c = self.cfg
        plans = self.plan_all(job, t)
        mask = np.zeros(self.A, dtype=bool)
        mind = 1e9
        for i, p in enumerate(plans):
            if p is not None:
                mask[i] = True
                mind = min(mind, p[1] - p[0])
        nh = math.floor(t) + 1.0
        if mask.any() and nh < c.H and (job.deadline - nh) >= mind and job.ndef < c.max_defer:
            mask[c.K + 1] = True
        ctx = Ctx()
        ctx.job, ctx.t, ctx.plans, ctx.mask, ctx.env, ctx.A = job, t, plans, mask, self, self.A
        return ctx

    def features(self, ctx):
        c, job, t = self.cfg, ctx.job, ctx.t
        i0 = min(int(t), self.T - 31)
        f = self.fc
        ci0 = max(self.ci[min(int(t), self.T - 1)], 1.0)
        tcl = c.t_cl0_s * 2.0 ** ((job.n - c.n_ref) / c.kappa) / 3600.0
        rem = job.deadline - t
        hod = t % 24
        x = np.zeros(47)
        x[0] = job.n / 50.0
        x[1] = job.layers / 6.0
        x[2] = job.kind
        x[3] = math.log1p(c.p_node_kw * tcl * c.pue) / 3.0
        x[4] = math.log1p(tcl) / 2.5
        x[5] = min(max(rem, 0.0), 36.0) / 24.0
        x[6] = job.iters / 300.0
        x[7] = job.cls / 2.0
        x[8] = ci0 / 500.0
        x[9] = f[i0 + 1:i0 + 7].mean() / ci0
        x[10] = f[i0 + 7:i0 + 25].mean() / ci0
        x[11] = f[i0:i0 + 25].min() / ci0
        x[12] = math.sin(2 * math.pi * hod / 24)
        x[13] = math.cos(2 * math.pi * hod / 24)
        x[14] = 1.0 if (int(t // 24) % 7) >= 5 else 0.0
        x[15] = (f[i0 + 3] - f[i0]) / ci0
        o = 16
        for i, p in enumerate(ctx.plans):
            if p is not None:
                x[o] = 1.0
                x[o + 1] = math.log1p(p[3]) / 3.0
                x[o + 2] = min(max(p[0] - t, 0.0), 36.0) / 6.0
                x[o + 3] = min(p[1] - p[0], 36.0) / 6.0
                x[o + 4] = p[2]
            o += 5
        for k in range(c.K):
            x[36 + 3 * k] = 0.0 if self.state[k] == "warm" else 1.0
            x[37 + 3 * k] = max(0.0, self.cold_at[k] - t) / 30.0 if self.state[k] == "cooling" else 0.0
            x[38 + 3 * k] = self.hsc(k, t) / 24.0
        x[45] = sum(1 for s in self.state if s != "warm") / c.K
        x[46] = min(self.trailing_util(t)[0], 3.0) / 3.0
        return x

    # ---------------- execution
    def execute(self, job, t, a, ctx):
        c = self.cfg
        K = c.K
        if a == K + 1:
            self.m["n_def"] += 1
            job.ndef += 1
            self.rewards.append(-c.lam_wait * (nh_event(t) - t))
            return nh_event(t)
        p = ctx.plans[a]
        s, e = p[0], p[1]
        d = e - s
        ci_true = self.ci_int(s, e)
        kg = (p[3] / d) * ci_true / 1000.0
        miss = e > job.deadline + 1e-9
        cost = kg + c.mu * p[3]
        if a == 0:
            heapq.heapreplace(self.nodes, e)
            self.m["n_cl"] += 1
        else:
            k = a - 1
            self.free_at[k] = e
            self.last_end[k] = e
            self._add_busy(s, e)
            self.m["n_q"] += 1
            self.m["qbusy"] += d
            self.m["per_unit"][k] += 1
            sh_kg = (c.p_fridge_kw + c.p_idle_kw) * ci_true / 1000.0
            cost += c.s_c * (sh_kg + c.mu * p[6])
        cost += c.zeta * miss + c.lam_wait * (s - t)
        self.m["op_kwh"] += p[3]
        self.m["op_kg"] += kg
        self.m["miss"] += int(miss)
        self.m["wait"] += (s - job.arr)
        self.m["jobs"] += 1
        self.rewards.append(-cost)
        return None

    def run(self, act, prov_pol=None):
        c = self.cfg
        ev = []
        seq = 0
        for j in self.jobs:
            heapq.heappush(ev, (j.arr, seq, 0, j))
            seq += 1
        for d in range(int(c.H // 24)):
            heapq.heappush(ev, (24.0 * d, seq, 1, None))
            seq += 1
        while ev:
            t, _, kind, job = heapq.heappop(ev)
            self.advance_fleet(t)
            if kind == 1:
                if self.prov == "ppo":
                    cc = self.cum_cost()
                    if self.prov_steps > 0:
                        self.prov_rewards.append(-(cc - self.cc_prev))
                    pa = prov_pol(self.prov_features(t), self.prov_mask(t))
                    self.apply_prov(pa, t)
                    self.cc_prev = cc
                    self.prov_steps += 1
                else:
                    self.provision(t)
                continue
            ctx = self.make_ctx(job, t)
            if not ctx.mask.any():
                self.m["unserved"] += 1
                continue
            a = act(ctx)
            self.m["decisions"] += 1
            r = self.execute(job, t, a, ctx)
            if r is not None:
                heapq.heappush(ev, (r, seq, 0, job))
                seq += 1
        self.advance_fleet(c.H)
        if self.prov == "ppo" and self.prov_steps > 0:
            self.prov_rewards.append(-(self.cum_cost() - self.cc_prev))
        return self.summary()

    def summary(self):
        m = self.m
        c = self.cfg
        n = len(self.jobs)
        tot_kwh = m["op_kwh"] + self.fleet_kwh
        tot_kg = m["op_kg"] + self.fleet_kg
        done = m["jobs"]
        viol = m["miss"] + m["unserved"]
        out = dict(
            n_jobs=n, completed=done, unserved=m["unserved"], miss=m["miss"],
            sla_viol=viol / max(n, 1), served_frac=done / max(n, 1),
            total_kwh=tot_kwh, total_kg=tot_kg, op_kwh=m["op_kwh"], op_kg=m["op_kg"],
            fleet_kwh=self.fleet_kwh, fleet_kg=self.fleet_kg,
            kwh_per_job=tot_kwh / max(done, 1), kg_per_job=tot_kg / max(done, 1),
            qpu_util=m["qbusy"] / max(self.cold_unit_h, 1e-9) if self.cold_unit_h > 0 else 0.0,
            frac_q=m["n_q"] / max(done, 1), n_def=m["n_def"], mean_wait=m["wait"] / max(done, 1),
            cooldowns=m["cooldowns"], gates=m["gates"], cold_unit_h=self.cold_unit_h,
            total_cost=tot_kg + c.mu * tot_kwh + c.zeta * viol + c.lam_wait * m["wait"],
            wait_h=m["wait"],
        )
        return out


def nh_event(t):
    return math.floor(t) + 1.0


# ------------------------------------------------------------------ policies
class ClassicalOnly:
    def __call__(self, ctx):
        return 0


class FidelityFirst:
    """Draft baseline: route every job to the lowest-noise feasible QPU regardless of energy/queue; classical only as fallback."""

    def __call__(self, ctx):
        for k in range(1, ctx.env.cfg.K + 1):
            if ctx.mask[k]:
                return k
        return 0


class FidelityFirstDL:
    """Stronger fidelity-first: lowest-noise QPU whose projected completion meets the deadline; else classical; else earliest finish."""

    def __call__(self, ctx):
        job = ctx.job
        for k in range(1, ctx.env.cfg.K + 1):
            p = ctx.plans[k]
            if p is not None and p[1] <= job.deadline:
                return k
        if ctx.plans[0] is not None and ctx.plans[0][1] <= job.deadline:
            return 0
        best, be = None, 1e18
        for i, p in enumerate(ctx.plans):
            if p is not None and p[1] < be:
                best, be = i, p[1]
        return best


def _best_now(ctx, tol=1e-9):
    job = ctx.job
    best, bc = None, 1e18
    for i, p in enumerate(ctx.plans):
        if p is not None and p[1] <= job.deadline + tol and p[5] < bc:
            best, bc = i, p[5]
    if best is None:  # nothing meets deadline: earliest completion
        be = 1e18
        for i, p in enumerate(ctx.plans):
            if p is not None and p[1] < be:
                best, be = i, p[1]
        bc = ctx.plans[best][5]
    return best, bc


class CostGreedy:
    def __call__(self, ctx):
        return _best_now(ctx)[0]


class CarbonDefer:
    """Cost-greedy + forecast-based temporal shifting inside the deadline slack."""

    def __init__(self, theta=0.05, margin=0.25, hmax=24):
        self.theta, self.margin, self.hmax = theta, margin, hmax

    def __call__(self, ctx):
        env, job, t = ctx.env, ctx.job, ctx.t
        a, c0 = _best_now(ctx)
        K = env.cfg.K
        if not ctx.mask[K + 1] or ctx.plans[a][1] > job.deadline:
            return a
        bf = c0
        nh = math.floor(t) + 1.0
        lw = env.cfg.lam_wait
        for h in range(0, self.hmax):
            tt = nh + h
            if tt >= env.cfg.H:
                break
            for p in env.plan_all(job, tt):
                if p is not None and p[1] <= job.deadline - self.margin and p[5] + lw * (tt - t) < bf:
                    bf = p[5] + lw * (tt - t)
        if bf < c0 * (1 - self.theta):
            return K + 1
        return a
