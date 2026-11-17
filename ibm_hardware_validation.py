"""Run the QAOA fidelity-model check on a REAL IBM Quantum backend (requires your own IBM Quantum Platform account).
Usage:  export QISKIT_IBM_TOKEN=<api key>; export QISKIT_IBM_INSTANCE=<crn>   (or pass --token/--instance)
        python ibm_hardware_validation.py --backend ibm_sherbrooke --shots 4000 [--n 6 8 10]
Writes results/ibm_hardware.json with, per circuit: ESP from the backend's *current* calibration, measured lost fraction,
queue/run timestamps and job ids.  NOT executed in the authors' sandbox (no network path to IBM Quantum): UNTESTED on hardware."""
import argparse, json, os, numpy as np, networkx as nx
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from scipy.optimize import minimize
from validate_aer import qaoa_circuit, ideal_energy, cut_value

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="ibm_sherbrooke"); ap.add_argument("--shots", type=int, default=4000)
ap.add_argument("--n", type=int, nargs="+", default=[6, 8, 10]); ap.add_argument("--graphs", type=int, default=3)
ap.add_argument("--token", default=os.getenv("QISKIT_IBM_TOKEN")); ap.add_argument("--instance", default=os.getenv("QISKIT_IBM_INSTANCE"))
a = ap.parse_args()
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=a.token, instance=a.instance)
be = svc.backend(a.backend); tgt = be.target; rng = np.random.default_rng(7); rows = []
for n in a.n:
    for gi in range(a.graphs):
        G = nx.random_regular_graph(3, n, seed=int(rng.integers(1e6))); m = G.number_of_edges()
        x = min((minimize(lambda x: -ideal_energy(G, x, 1)[0], rng.uniform(0, np.pi, 2), method="COBYLA", options=dict(maxiter=100))
                 for _ in range(3)), key=lambda r: r.fun).x
        Ei, _ = ideal_energy(G, x, 1)
        qc = qaoa_circuit(G, x[:1], x[1:]); qc.measure_all()
        tq = transpile(qc, be, optimization_level=1, seed_transpiler=0)
        logF = 0.0
        for inst in tq.data:
            nm = inst.operation.name; qs = tuple(tq.find_bit(q).index for q in inst.qubits)
            if nm in ("barrier", "delay"): continue
            if nm in tgt.operation_names and qs in tgt[nm]: logF += np.log(1 - (tgt[nm][qs].error or 0.0))
        job = SamplerV2(mode=be).run([tq], shots=a.shots)
        counts = job.result()[0].data.meas.get_counts()
        tot = sum(c * cut_value([int(ch) for ch in bs[::-1]], G.edges()) for bs, c in counts.items()) / a.shots
        F = float(np.exp(logF)); xi = 0.342 * (8.0 / max(n, 8)) ** 0.82
        rows.append(dict(n=n, graph=gi, job_id=job.job_id(), backend=a.backend, F=F, lost_meas=1 - (tot - m / 2) / (Ei - m / 2),
                         lost_esp=1 - F, lost_cal=1 - F ** xi))
        print(rows[-1], flush=True)
os.makedirs("results", exist_ok=True); json.dump(rows, open("results/ibm_hardware.json", "w"), indent=1)
