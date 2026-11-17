"""
EcoQ hybrid quantum-classical data-center simulator.

Sequential list-scheduling simulator (decision at each job arrival, back-filling
timelines per resource).  All hardware / grid parameters are ASSUMPTIONS that are
listed in PARAMS and varied in the sensitivity analysis; nothing here is measured
from a physical facility.
"""
import math, bisect
import numpy as np

H, DAY = 3600.0, 86400.0

# --------------------------------------------------------------------------- #
# Parameters (all ASSUMPTIONS; see paper Table of parameters)
# --------------------------------------------------------------------------- #
QPUS = [
    dict(name="SC-A", nq=27, e_floor2=0.0030, e1=3e-4, ero=0.012, t1q=40e-9, t2q=400e-9,
         t_ro=1e-6, t_reset=250e-6, swap=1.8, par=("frac", 0.5), T2=100e-6,
         P_fix=15e3, P_dyn=2.5e3, phase=2 * H),
    dict(name="SC-B", nq=65, e_floor2=0.0050, e1=5e-4, ero=0.025, t1q=50e-9, t2q=500e-9,
         t_ro=1e-6, t_reset=250e-6, swap=1.8, par=("frac", 0.5), T2=60e-6,
         P_fix=25e3, P_dyn=4e3, phase=10 * H),
    dict(name="TI", nq=32, e_floor2=0.0030, e1=1e-4, ero=0.005, t1q=10e-6, t2q=200e-6,
         t_ro=300e-6, t_reset=5e-3, swap=1.0, par=("abs", 2), T2=1.0,
         P_fix=5e3, P_dyn=3e3, phase=18 * H),
]
K = len(QPUS)

DEFAULTS = dict(
    horizon_h=168.0, lam_per_h=2.0, theta=0.35, regime="solar",
    n_nodes=4, P_idle=800.0, P_busy=3500.0, P_host=250.0, pue=1.25,
    fix_scale=1.0,            # scales all QPU fixed (cryo+control-idle) power
    cal_dur=2400.0, tau_setup=600.0, tau_idle=1800.0,
    drift_eta=0.3, sigma_kappa=0.25, sigma_tel=0.05,
    sigma_pred=0.3,           # profiling error on classical runtime (log-normal)
    f_s0=0.03, f_s1=0.004,    # carbon-forecast error: sigma = s0 + s1*lead_h
    xi_mode="esp",         # 'cal': Aer-calibrated energy-damping exponent for QAOA; 'esp': xi=1 everywhere
    setup_aware=True,       # whether predictions include session set-up energy
    rho_mit=0.6, mit_shots=3.0, mit_time=2.0, mit_floor=0.15,
    t_pre=10.0, t_iter=0.2,
    t_base=dict(qaoa=10.0, vqe=60.0, qml=20.0),
    slack_s=[1 * H, 6 * H, 24 * H, 72 * H], slack_p=[0.15, 0.35, 0.35, 0.15],
    w=dict(alpha=0.5, beta=0.5, gamma=0.3, lam=2.0, delta=1.0),
    E0=1.0, C0=300.0,         # reward normalisers (set after calibration)
)

KINDS = ["qaoa", "vqe", "qml"]
N_ACT = 2 + 2 * 2 * K  # classical now/defer + K qpus * mit(2) * timing(2)


def act_decode(a):
    """-> (target, mit, defer); target=-1 classical."""
    if a < 2:
        return -1, 0, a
    b = a - 2
    defer = b % 2
    km = b // 2
    return km // 2, km % 2, defer


# --------------------------------------------------------------------------- #
# Carbon-intensity traces: real 2025 hourly profiles (data/traces_2025.csv) + legacy synthetic regimes
# --------------------------------------------------------------------------- #
ZONES = ["US-CAL-CISO", "GB", "DE", "IN-WE", "FR"]
_REAL = None
# chronological split: windows starting in [0,TRAIN_END] (hours 0..TRAIN_END+n_h) for training/validation,
# windows starting in [TEST_START, 8760-n_h] for testing -> no hour is shared between the two sets.
TRAIN_START_MAX = 3000
TEST_START_MIN = 4500


def real_trace(zone, n_h, rng, split="test"):
    global _REAL
    if _REAL is None:
        import os, pandas as pd
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "traces_2025.csv")
        _REAL = pd.read_csv(p)
    x = _REAL[zone].to_numpy(float)
    hi_max = len(x) - n_h
    lo, hi = (0, TRAIN_START_MAX) if split == "train" else (TEST_START_MIN, hi_max)
    st = int(rng.integers(lo, hi + 1))
    return x[st:st + n_h].copy()


def make_trace(regime, n_h, rng, split="test"):
    if regime in ZONES:
        return real_trace(regime, n_h, rng, split)
    h = np.arange(n_h)
    hod = h % 24

    def ar1(phi, sd):
        x = np.zeros(n_h)
        s = sd * math.sqrt(1 - phi ** 2)
        x[0] = rng.normal(0, sd)
        for i in range(1, n_h):
            x[i] = phi * x[i - 1] + rng.normal(0, s)
        return x

    if regime == "solar":
        base = 330 - 200 * np.clip(np.sin(math.pi * (hod - 6) / 12), 0, None) ** 1.5
        I = base + ar1(0.9, 25)
        lo, hi = 40, 700
    elif regime == "wind":
        I = 300 + 30 * np.cos(2 * math.pi * (hod - 19) / 24) + ar1(0.985, 90)
        lo, hi = 60, 700
    elif regime == "fossil":
        I = 620 + 40 * np.sin(2 * math.pi * (hod - 14) / 24) + ar1(0.9, 20)
        lo, hi = 450, 800
    else:
        raise ValueError(regime)
    return np.clip(I, lo, hi)


# --------------------------------------------------------------------------- #
class Timeline:
    __slots__ = ("iv",)

    def __init__(self, blocks=()):
        self.iv = sorted(blocks)

    def earliest(self, r, d):
        t = r
        for s, e in self.iv:
            if e <= t:
                continue
            if s >= t + d:
                break
            t = e
        return t

    def prev_end(self, s):
        """end time of the latest interval finishing at or before s (or -inf)."""
        best = -1e18
        for a, b in self.iv:
            if b <= s and b > best:
                best = b
            if a >= s:
                break
        return best

    def insert(self, s, e):
        bisect.insort(self.iv, (s, e))


class Job:
    __slots__ = ("id", "kind", "n", "layers", "g2", "g1", "M", "I", "N0", "tol", "t_cl",
                 "t_cl_hat", "arr", "slack", "deadline")


def gen_jobs(rng, cfg):
    jobs, t, i = [], 0.0, 0
    T = cfg["horizon_h"] * H
    lam = cfg["lam_per_h"] / H
    while True:
        hod = (t / H) % 24
        rate = lam * (1 + 0.6 * math.sin(2 * math.pi * (hod - 9) / 24))
        t += rng.exponential(1.0 / (lam * 1.6))
        if t >= T:
            break
        if rng.random() > rate / (lam * 1.6):
            continue
        j = Job()
        j.id, j.arr = i, t
        i += 1
        j.kind = rng.choice(KINDS, p=[0.55, 0.30, 0.15])
        if j.kind == "qaoa":
            j.n = int(rng.integers(10, 25)); j.layers = int(rng.choice([1, 2, 3], p=[.4, .4, .2]))
            j.g2 = 3 * j.n * j.layers; j.g1 = j.n * (1 + 2 * j.layers)
            j.M, j.I, j.N0 = 1, int(rng.integers(60, 201)), int(rng.choice([2048, 4096, 8192]))
        elif j.kind == "vqe":
            j.n = int(rng.integers(6, 25)); j.layers = int(rng.integers(2, 7))
            j.g2 = j.layers * (j.n - 1); j.g1 = 2 * j.n * j.layers
            j.M = int(np.clip(math.ceil(0.25 * j.n ** 2), 10, 150))
            j.I, j.N0 = int(rng.integers(50, 201)), int(rng.choice([1024, 2048]))
        else:
            j.n = int(rng.integers(6, 17)); j.layers = int(rng.integers(3, 7))
            j.g2 = j.layers * j.n; j.g1 = 2 * j.n * j.layers
            j.M, j.I, j.N0 = 16, int(rng.integers(100, 301)), int(rng.choice([512, 1024]))
        j.tol = float(rng.uniform(0.10, 0.50))
        tcl = cfg["t_base"][j.kind] * math.exp(cfg["theta"] * (j.n - 10)) * math.exp(rng.normal(0, 0.6))
        j.t_cl = float(min(tcl, 3e5))
        j.t_cl_hat = j.t_cl * math.exp(rng.normal(0, cfg["sigma_pred"]))
        j.slack = float(rng.choice(cfg["slack_s"], p=cfg["slack_p"]))
        j.deadline = j.arr + j.slack
        jobs.append(j)
    return jobs


# --------------------------------------------------------------------------- #
class Env:
    """One episode = one simulated week.  step(a) schedules the current job."""

    def __init__(self, cfg=None, ablate=None):
        self.cfg0 = {**DEFAULTS, **(cfg or {})}
        self.ablate = set(ablate or [])
        self.obs_dim = None

    # ------------------------------------------------------------------ reset
    def reset(self, seed, **over):
        cfg = {**self.cfg0, **over}
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.T = cfg["horizon_h"] * H
        self.nH = int(24 * 60)
        self.I = make_trace(cfg["regime"], self.nH, rng, cfg.get("trace_split", "test"))
        self.cumT = np.concatenate([[0], np.cumsum(self.I * H)])  # g/kWh * s
        self.eps = rng.normal(0, 1, self.nH)                      # fixed forecast noise
        self.kappa = [np.exp(rng.normal(0, cfg["sigma_kappa"], 400)) for _ in range(K)]
        self.kappa_tel = [np.exp(rng.normal(0, cfg["sigma_tel"], 400)) for _ in range(K)]
        self.jobs = gen_jobs(rng, cfg)
        self.cls = [Timeline() for _ in range(cfg["n_nodes"])]
        self.qtl = []
        for k in range(K):
            ph = QPUS[k]["phase"]
            blocks = [(ph + d * DAY, ph + d * DAY + cfg["cal_dur"]) for d in range(0, 120)]
            self.qtl.append(Timeline(blocks))
        self.qjobs = [[] for _ in range(K)]      # (start,end,ctrl_s,P_dyn_energy parts)
        self.cjobs = []                          # classical dynamic intervals (s,e)
        self.rec = []                            # per-job records
        self.idx = 0
        self.E_dyn = 0.0; self.C_dyn = 0.0
        self.last_end = 0.0
        self._prep()
        return self._obs()

    # ------------------------------------------------------------- carbon util
    def cum(self, t, cumarr):
        t = max(t, 0.0)
        h = int(t // H)
        h = min(h, len(cumarr) - 2)
        return cumarr[h] + (cumarr[h + 1] - cumarr[h]) * ((t - h * H) / H)

    def mean_I(self, t0, t1, cumarr):
        if t1 - t0 < 1e-6:
            return float(cumarr[min(int(t0 // H), len(cumarr) - 2) + 1] - cumarr[min(int(t0 // H), len(cumarr) - 2)]) / H
        return (self.cum(t1, cumarr) - self.cum(t0, cumarr)) / (t1 - t0)

    def _forecast_cum(self, now):
        c = self.cfg
        h0 = int(now // H)
        h = np.arange(self.nH)
        lead = np.clip(h - h0, 0, None)
        perfect = "perfect_fc" in self.ablate
        sig = (c["f_s0"] + c["f_s1"] * np.minimum(lead, 72)) * (0.0 if perfect else 1.0)   # error saturates at 72 h lead
        F = self.I * np.exp(self.eps * sig - 0.5 * sig ** 2)
        if not perfect:
            F[h > h0 + 72] = self.I.mean()        # beyond 72 h: climatological mean (no skill)
        F[h < h0] = self.I[h < h0]
        return np.concatenate([[0], np.cumsum(F * H)]), F

    # --------------------------------------------------------- QPU physics
    def qpu_mult(self, k, t, truth):
        c = self.cfg
        q = QPUS[k]
        cyc = int(math.floor((t - q["phase"] - c["cal_dur"]) / DAY))
        age = t - (q["phase"] + c["cal_dur"] + cyc * DAY)
        cyc = min(cyc, 390)
        base = 1.0 + c["drift_eta"] * age / DAY
        if truth or "oracle" in self.ablate:
            kap = self.kappa[k][cyc + 1]
        else:
            now_cyc = int(math.floor((self.now - q["phase"] - c["cal_dur"]) / DAY))
            kap = self.kappa[k][cyc + 1] * self.kappa_tel[k][cyc + 1] if cyc == now_cyc else 1.0
        return kap * base

    def qpu_circ(self, k, j, mit, t, truth):
        """fidelity F, mitigated bias, shot time, total shots for job j on QPU k at time t."""
        q, c = QPUS[k], self.cfg
        m = self.qpu_mult(k, t, truth)
        g2 = j.g2 * q["swap"]
        e2 = min((q["e_floor2"] + q["t2q"] / q["T2"]) * m, 0.5)
        e1 = min(q["e1"] * m, 0.5); ero = min(q["ero"] * m, 0.5)
        lnF = g2 * math.log1p(-e2) + j.g1 * math.log1p(-e1) + j.n * math.log1p(-ero)
        F = math.exp(lnF)
        # energy-observable damping exponent: lost fraction of the ideal-to-mixed gap = 1 - F**xi
        # QAOA: power law fitted to Qiskit-Aer data for n in [8,12] (held constant beyond); others: xi=1 (ESP)
        if c["xi_mode"] == "cal" and j.kind == "qaoa":
            xi = 0.342 * (8.0 / min(max(j.n, 8), 12)) ** 0.82
        else:
            xi = 1.0
        lost = 1 - F ** xi
        if mit:
            rho = c["rho_mit"] if F >= c["mit_floor"] else 0.0
            bias = lost * (1 - rho)
        else:
            bias = lost
        par = q["par"]
        p = par[1] * j.n if par[0] == "frac" else par[1]
        depth2 = g2 / max(p, 1.0)
        tau = depth2 * q["t2q"] + (2 * j.layers + 2) * q["t1q"] + q["t_ro"]
        if mit:
            tau *= c["mit_time"]
        shots = j.I * j.M * j.N0 * (c["mit_shots"] if mit else 1.0)
        t_shot = tau + q["t_reset"]
        return F, bias, t_shot, shots

    # ------------------------------------------------------------- planning
    def _plan(self, j, tgt, mit, release, truth, cumarr):
        """Place job on resource; return dict with start,end,E(kWh),C(g),ok."""
        c = self.cfg
        pue = c["pue"]
        if tgt < 0:
            d = j.t_cl if truth else j.t_cl_hat
            best = None
            for tl in self.cls:
                s = tl.earliest(release, d)
                if best is None or s < best[0]:
                    best = (s, tl)
            s = best[0]
            E = (c["P_busy"] - c["P_idle"]) * pue * d / 3.6e6
            C = E * self.mean_I(s, s + d, cumarr)
            return dict(tgt=-1, start=s, end=s + d, E=E, C=C, ok=True, bias=0.0, dur=d, tl=best[1],
                        setup=False, ctrl=0.0, sE=0.0, sC=0.0)
        q = QPUS[tgt]
        # duration is time-independent to first order -> compute at release then re-evaluate at start
        F, bias, t_shot, shots = self.qpu_circ(tgt, j, mit, release, truth)
        dur = c["t_pre"] + shots * t_shot + j.I * c["t_iter"]
        s = self.qtl[tgt].earliest(release, dur)
        F, bias, t_shot, shots = self.qpu_circ(tgt, j, mit, s, truth)
        dur = c["t_pre"] + shots * t_shot + j.I * c["t_iter"]
        s = self.qtl[tgt].earliest(s, dur)
        ctrl = shots * t_shot
        pe = self.qtl[tgt].prev_end(s)
        setup = (s - pe) > c["tau_idle"]
        P = q["P_dyn"]
        E = (P * ctrl + c["P_host"] * dur) * pue / 3.6e6
        C = E * self.mean_I(s, s + dur, cumarr)
        sE = sC = 0.0
        if setup and (truth or self.cfg.get("setup_aware", True)):
            sE = P * c["tau_setup"] * pue / 3.6e6
            sC = sE * self.mean_I(max(s - c["tau_setup"], 0), s, cumarr)
            E += sE
            C += sC
        ok = bias <= j.tol
        return dict(tgt=tgt, start=s, end=s + dur, E=E, C=C, ok=ok, bias=bias, dur=dur, setup=setup,
                    ctrl=ctrl, F=F, sE=sE, sC=sC)

    def _release_defer(self, j, dur, cumF):
        """Forecast-greenest release within slack window; None if no window."""
        now = self.now
        latest = j.deadline - 1.15 * dur - 1800.0
        latest = min(latest, now + 72 * H)
        if latest - now < 1800.0:
            return None
        cand = np.arange(now + 900.0, latest, 900.0)
        if len(cand) == 0:
            return None
        hs = np.clip(cand // H, 0, len(cumF) - 2).astype(int)
        cs = cumF[hs] + (cumF[hs + 1] - cumF[hs]) * ((cand - hs * H) / H)
        e = cand + dur
        he = np.clip(e // H, 0, len(cumF) - 2).astype(int)
        ce = cumF[he] + (cumF[he + 1] - cumF[he]) * ((e - he * H) / H)
        m = (ce - cs) / dur
        return float(cand[int(np.argmin(m))])

    # -------------------------------------------------------------- features
    def _prep(self):
        if self.idx >= len(self.jobs):
            self.done = True
            self.now = self.T
            return
        self.done = False
        j = self.jobs[self.idx]
        self.now = j.arr
        c = self.cfg
        cumF, F = self._forecast_cum(self.now)
        self.cumF = cumF
        plans = [None] * N_ACT
        mask = np.zeros(N_ACT, dtype=bool)
        # classical
        for dfr in (0, 1):
            rel = self.now
            if dfr:
                rel = self._release_defer(j, j.t_cl_hat, cumF)
                if rel is None:
                    continue
            p = self._plan(j, -1, 0, rel, False, cumF)
            plans[dfr] = p; mask[dfr] = True
        for k in range(K):
            if j.n > QPUS[k]["nq"]:
                continue
            for mit in (0, 1):
                for dfr in (0, 1):
                    a = 2 + (k * 2 + mit) * 2 + dfr
                    rel = self.now
                    if dfr:
                        _, _, ts, sh = self.qpu_circ(k, j, mit, self.now, False)
                        dur = c["t_pre"] + sh * ts + j.I * c["t_iter"]
                        rel = self._release_defer(j, dur, cumF)
                        if rel is None:
                            continue
                    p = self._plan(j, k, mit, rel, False, cumF)
                    plans[a] = p
                    mask[a] = (p["bias"] <= j.tol) if "no_shield" not in self.ablate else True
        if "no_defer" in self.ablate:
            mask[1::2] = False; mask[0] = True
            # odd indices are defer actions (classical-defer idx1, qpu defer odd)
        if "no_mit" in self.ablate:
            for k in range(K):
                for dfr in (0, 1):
                    mask[2 + (k * 2 + 1) * 2 + dfr] = False
        self.plans, self.mask = plans, mask
        w = c["w"]; bc = 1e18
        for a in range(N_ACT):
            p = plans[a]
            if p is None or not mask[a]:
                continue
            lat = p["end"] - j.arr
            cc = (w["alpha"] * p["E"] / c["E0"] + w["beta"] * p["C"] / c["C0"]
                  + w["gamma"] * min(lat / j.slack, 3.0) + w["lam"] * float(p["end"] > j.deadline))
            bc = min(bc, cc)
        self.base_cost = bc
        self.F_now = F

    def _obs(self):
        if self.done:
            return np.zeros(self.obs_dim or 1, dtype=np.float32), None
        j = self.jobs[self.idx]
        c = self.cfg
        now = self.now
        I_now = self.I[int(now // H)]
        h0 = int(now // H)
        Fz = self.F_now
        f6, f12, f24 = Fz[h0:h0 + 6].mean(), Fz[h0:h0 + 12].mean(), Fz[h0:h0 + 24].mean()
        lim = min(h0 + max(int(j.slack // H), 1), self.nH - 1)
        win = Fz[h0:lim + 1]
        fmin = win.min(); tmin = (int(np.argmin(win))) * H / max(j.slack, H)
        kind = [float(j.kind == k) for k in KINDS]
        hod = (now / H) % 24
        f = kind + [j.n / 24, (math.log10(j.t_cl_hat) - 2) / 3, (math.log10(j.slack / H)) / 1.5,
                    j.tol * 2, math.sin(2 * math.pi * hod / 24), math.cos(2 * math.pi * hod / 24)]
        if "no_carbon" in self.ablate:
            carb = [0.0] * 6
        else:
            carb = [I_now / 500, f6 / 500, f12 / 500, f24 / 500, fmin / 500, min(tmin, 3.0)]
        f += carb
        busy = sum(1 for tl in self.cls if tl.earliest(now, 1.0) > now) / len(self.cls)
        f += [busy]
        sl = max(j.slack, 1.0)
        for a in range(N_ACT):
            p = self.plans[a]
            if p is None:
                f += [0.0] * 6; continue
            if "no_qpu_state" in self.ablate and p["tgt"] >= 0:
                f += [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; continue
            wait = (p["start"] - now) / sl
            f += [1.0, min(wait, 3.0), math.log10(p["E"] + 1e-4) / 2 + 1,
                  (math.log10(p["C"] + 1e-1) - 1) / 3,
                  min((p["end"] - j.arr) / sl, 3.0),
                  (min(p["bias"], 2.0) - j.tol) if p["tgt"] >= 0 else 0.0]
        v = np.asarray(f, dtype=np.float32)
        if "no_carbon" in self.ablate:
            # also hide carbon-dependent plan features
            pass
        self.obs_dim = len(v)
        return v, self.mask.copy()

    # ----------------------------------------------------------------- step
    def step(self, a):
        j = self.jobs[self.idx]
        c, w = self.cfg, self.cfg["w"]
        tgt, mit, dfr = act_decode(a)
        pl = self.plans[a]
        assert pl is not None
        # re-plan with truth at the same release as the predicted plan
        if dfr:
            if tgt < 0:
                rel = self._release_defer(j, j.t_cl_hat, self.cumF)
            else:
                _, _, ts, sh = self.qpu_circ(tgt, j, mit, self.now, False)
                rel = self._release_defer(j, c["t_pre"] + sh * ts + j.I * c["t_iter"], self.cumF)
        else:
            rel = self.now
        tp = self._plan(j, tgt, mit, rel, True, self.cumT)
        E, C = tp["E"], tp["C"]
        rework = 0
        end = tp["end"]
        if tgt < 0:
            tp["tl"].insert(tp["start"], tp["end"])
            self.cjobs.append((tp["start"], tp["end"]))
        else:
            self.qtl[tgt].insert(tp["start"], tp["end"])
            self.qjobs[tgt].append((tp["start"], tp["end"], tp["ctrl"]))
            if not tp["ok"]:
                rework = 1
                fb = self._plan(j, -1, 0, tp["end"], True, self.cumT)
                fb["tl"].insert(fb["start"], fb["end"]); self.cjobs.append((fb["start"], fb["end"]))
                E += fb["E"]; C += fb["C"]; end = fb["end"]
        lat = end - j.arr
        miss = int(end > j.deadline)
        cost = (w["alpha"] * E / c["E0"] + w["beta"] * C / c["C0"] + w["gamma"] * min(lat / j.slack, 3.0)
                + w["lam"] * miss + w["delta"] * rework)
        self.E_dyn += E; self.C_dyn += C; self.last_end = max(self.last_end, end)
        self.rec.append(dict(id=j.id, kind=j.kind, n=j.n, tgt=tgt, mit=mit, defer=dfr, E=E, C=C, lat=lat,
                             miss=miss, rework=rework, slack=j.slack, setup=tp["setup"], cost=cost,
                             sE=tp["sE"], sC=tp["sC"]))
        base = self.base_cost
        self.idx += 1
        self._prep()
        obs = self._obs()
        return obs, -cost, self.done, dict(cost=cost, base=base)

    # --------------------------------------------------------- finalisation
    def summary(self):
        c = self.cfg
        pue = c["pue"]
        Tend = self.T   # steady-state week: fixed loads accounted over the horizon only
        drain_h = max(0.0, self.last_end - self.T) / H
        # exact setup accounting for QPUs (sessions); replaces per-decision estimates
        E_setup = C_setup = E_ctrl = 0.0
        busy = []
        n_cal = Tend / DAY
        for k in range(K):
            prev, bsy = -1e18, 0.0
            for (s_, e_, ctrl) in sorted(self.qjobs[k]):
                if s_ - prev > c["tau_idle"] and not self._near_cal(k, s_):
                    Es = QPUS[k]["P_dyn"] * c["tau_setup"] * pue / 3.6e6
                    E_setup += Es
                    C_setup += Es * self.mean_I(max(s_ - c["tau_setup"], 0), s_, self.cumT)
                prev = e_; bsy += max(0.0, min(e_, Tend) - min(s_, Tend))
                E_ctrl += QPUS[k]["P_dyn"] * ctrl * pue / 3.6e6
            busy.append(bsy / (Tend - n_cal * c["cal_dur"]))
        est_E = sum(r["sE"] for r in self.rec); est_C = sum(r["sC"] for r in self.rec)
        E_dyn = self.E_dyn - est_E + E_setup
        C_dyn = self.C_dyn - est_C + C_setup
        # fixed loads
        P_fix_q = sum(q["P_fix"] for q in QPUS) * c["fix_scale"]
        n_days = Tend / DAY
        P_cal = sum(q["P_dyn"] for q in QPUS) * (c["cal_dur"] / DAY)  # average W for calibration
        P_cls_idle = c["n_nodes"] * c["P_idle"]
        P_fix = (P_fix_q + P_cls_idle + P_cal) * pue
        E_fix = P_fix * Tend / 3.6e6
        C_fix = P_fix / 3.6e6 * self.cum(Tend, self.cumT)
        n = len(self.rec)
        lat = np.array([r["lat"] for r in self.rec])
        sl = np.array([r["slack"] for r in self.rec])
        tg = np.array([r["tgt"] for r in self.rec])
        qfix_share = (P_fix_q * pue * Tend / 3.6e6)
        return dict(
            n_jobs=n, T_end_h=Tend / H, drain_h=drain_h,
            E_total=E_dyn + E_fix, E_dyn=E_dyn, E_fix=E_fix, E_qfix=qfix_share, E_ctrl=E_ctrl, E_setup=E_setup,
            C_total=(C_dyn + C_fix) / 1e3, C_dyn=C_dyn / 1e3, C_fix=C_fix / 1e3,
            slo_miss=float(np.mean([r["miss"] for r in self.rec])),
            rework=float(np.mean([r["rework"] for r in self.rec])),
            offload=float(np.mean(tg >= 0)),
            defer=float(np.mean([r["defer"] for r in self.rec])),
            mit=float(np.mean([r["mit"] for r in self.rec if r["tgt"] >= 0] or [0.0])),
            lat_h=float(lat.mean() / H), slow=float(np.mean(lat / sl)),
            util_qpu=float(np.mean(busy)), util_each=[float(x) for x in busy],
            E_per_job=(E_dyn + E_fix) / n, Edyn_per_job=E_dyn / n,
            cost=float(sum(r["cost"] for r in self.rec)),
            share_k=[float(np.mean(tg == k)) for k in range(K)],
        )

    def _near_cal(self, k, s):
        q = QPUS[k]; c = self.cfg
        cyc = math.floor((s - q["phase"] - c["cal_dur"]) / DAY)
        end = q["phase"] + c["cal_dur"] + cyc * DAY
        return 0 <= s - end <= c["tau_idle"]
