"""Out-of-sample check of the calibrated fidelity model on a REAL IBM device calibration snapshot.
Noise = per-qubit/per-gate errors, T1/T2 and readout errors of ibm_sherbrooke (127q heavy-hex, Eagle) as shipped in
qiskit_ibm_runtime.fake_provider (a stored calibration snapshot; execution is Aer noisy simulation, NOT hardware).
Model under test: F = prod(1-e_g) from the snapshot's actual gate/readout errors; lost = 1-F^xi with xi = 0.342*(8/n)^0.82
(fitted earlier on uniform depolarising noise, used here WITHOUT refitting). Output: results/aer_ibm_calib.json"""
import json, numpy as np, networkx as nx
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from scipy.optimize import minimize
from validate_aer import qaoa_circuit, ideal_energy, cut_value

rng = np.random.default_rng(7)
be = FakeSherbrooke(); tgt = be.target
sim = AerSimulator.from_backend(be, method="density_matrix")
rows = []
for n in (6, 8, 10):
    for gi in range(3):
        G = nx.random_regular_graph(3, n, seed=int(rng.integers(1e6))); m = G.number_of_edges()
        x = min((minimize(lambda x: -ideal_energy(G, x, 1)[0], rng.uniform(0, np.pi, 2), method="COBYLA", options=dict(maxiter=100))
                 for _ in range(3)), key=lambda r: r.fun).x
        Ei, cuts = ideal_energy(G, x, 1)
        qc = qaoa_circuit(G, x[:1], x[1:]); qc.measure_all()
        for seed_t in range(2):                       # different layouts -> different qubits/couplers
            tq = transpile(qc, be, optimization_level=1, seed_transpiler=seed_t)
            lay = tq.layout.final_index_layout() if tq.layout is not None else None
            logF = 0.0
            for inst in tq.data:
                nm = inst.operation.name
                qs = tuple(tq.find_bit(q).index for q in inst.qubits)
                if nm in ("barrier", "delay") : continue
                if nm == "measure":
                    pr = tgt["measure"][qs].error; logF += np.log(1 - (pr or 0)); continue
                if nm in tgt.operation_names and qs in tgt[nm]:
                    e = tgt[nm][qs].error or 0.0; logF += np.log(1 - e)
            F = float(np.exp(logF))
            counts = sim.run(tq, shots=8000, seed_simulator=int(rng.integers(1e6))).result().get_counts()
            tot = sum(c * cut_value([int(ch) for ch in bs[::-1]], G.edges()) for bs, c in counts.items()) / 8000
            lost = 1 - (tot - m / 2) / (Ei - m / 2)
            xi = 0.342 * (8.0 / n) ** 0.82
            rows.append(dict(n=n, graph=gi, layout=seed_t, F=F, lost_aer=float(lost), lost_esp=1 - F,
                             lost_cal=1 - F ** xi, cx=int(tq.count_ops().get("ecr", 0)), depth=tq.depth()))
    print("n", n, "done", flush=True)
a = np.array([r["lost_aer"] for r in rows]); e = np.array([r["lost_esp"] for r in rows]); c = np.array([r["lost_cal"] for r in rows])
S = dict(n_points=len(rows), device="ibm_sherbrooke calibration snapshot (fake_provider)",
         mae_esp=float(np.abs(e - a).mean()), bias_esp=float((e - a).mean()),
         mae_cal=float(np.abs(c - a).mean()), bias_cal=float((c - a).mean()),
         optimistic_gt_0p05=float(np.mean(c - a < -0.05)), pearson_cal=float(np.corrcoef(c, a)[0, 1]),
         median_ecr=float(np.median([r["cx"] for r in rows])))
json.dump(dict(summary=S, points=rows), open("results/aer_ibm_calib.json", "w"), indent=1)
print(json.dumps(S, indent=1))
