"""Extended validation of the fidelity model.
(A) VQE-type workloads: hardware-efficient ansatz, random molecular-like Pauli-sum Hamiltonians, exact density-matrix simulation
    with depolarising noise  -> xi_VQE fitted on Hamiltonian 0-1, tested on Hamiltonian 2.
(B) Realistic noise: QAOA circuits under depolarising floor + thermal relaxation (T1=T2) -> held-out error of the
    simulator's calibrated model (eps2 = floor + t2q/T2, xi(n) power law) without refitting."""
import json, itertools, numpy as np, networkx as nx
from scipy.optimize import minimize, minimize_scalar
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
rng = np.random.default_rng(4242)

def ansatz(n, L, th):
    qc = QuantumCircuit(n); k = 0
    for _ in range(L):
        for q in range(n):
            qc.ry(th[k], q); qc.rz(th[k + 1], q); k += 2
        for q in range(n - 1): qc.cx(q, q + 1)
    return qc

def rand_H(n):
    terms = []
    for i in range(n): terms.append(("Z", [i], rng.normal(0, 1.0)))
    for i in range(n - 1):
        for P in ("ZZ", "XX", "YY"): terms.append((P, [i, i + 1], rng.normal(0, 0.5)))
    for _ in range(6):
        idx = sorted(rng.choice(n, 4, replace=False)); terms.append((rng.choice(["ZZZZ", "XXYY", "YYXX"]), list(map(int, idx)), rng.normal(0, 0.2)))
    return SparsePauliOp.from_sparse_list([(p, q, c) for p, q, c in terms], num_qubits=n)

def lost(Eno, Eid, Emix): return 1 - (Eno - Emix) / (Eid - Emix)

def noisy_energy(qc, H, nm, basis):
    sim = AerSimulator(method="density_matrix", noise_model=nm, basis_gates=basis)
    c = transpile(qc, sim, basis_gates=basis, optimization_level=0); c.save_density_matrix()
    rho = sim.run(c).result().data()["density_matrix"]
    return float(np.real(DensityMatrix(rho).expectation_value(H))), c.count_ops()

# ------------------------------------------------------------ (A) VQE-type
BAS = ["cx", "ry", "rz", "rx", "h", "id"]
A = []
for n in (4, 6, 8):
    for hi in range(3):
        H = rand_H(n); Emix = float(np.real(H.to_matrix(sparse=True).diagonal().sum()) / 2 ** n)
        for L in (1, 2, 3):
            f = lambda th: float(np.real(Statevector(ansatz(n, L, th)).expectation_value(H)))
            best = min((minimize(f, rng.uniform(0, 2 * np.pi, 2 * n * L), method="COBYLA", options=dict(maxiter=250)) for _ in range(2)), key=lambda r: r.fun)
            Eid = best.fun
            if Emix - Eid < 1e-3: continue
            qc = ansatz(n, L, best.x)
            for e2 in (0.005, 0.01, 0.02, 0.04):
                e1 = e2 / 10
                nm = NoiseModel(); nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["cx"])
                nm.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ["ry", "rz"])
                En, ops = noisy_energy(qc, H, nm, BAS)
                g2 = ops.get("cx", 0); g1 = ops.get("ry", 0) + ops.get("rz", 0)
                F = (1 - e2) ** g2 * (1 - e1) ** g1
                A.append(dict(n=n, ham=hi, L=L, e2=e2, F=F, lost=lost(En, Eid, Emix)))
    print("VQE n", n, "done", len(A), flush=True)
def fit(rows, key="F", lk="lost"):
    return minimize_scalar(lambda x: np.mean([(1 - r[key] ** x - r[lk]) ** 2 for r in rows]), bounds=(0.05, 3), method="bounded").x
tr = [r for r in A if r["ham"] in (0, 1)]; te = [r for r in A if r["ham"] == 2]
xi_v = fit(tr)
mae = lambda rows, x: float(np.mean([abs(1 - r["F"] ** x - r["lost"]) for r in rows]))
resA = dict(n_points=len(A), xi_train=float(xi_v), xi_all=float(fit(A)), mae_test_xi1=mae(te, 1.0), mae_test_cal=mae(te, xi_v),
            bias_test_xi1=float(np.mean([(1 - r["F"]) - r["lost"] for r in te])),
            frac_xi1_conservative=float(np.mean([(1 - r["F"]) >= r["lost"] for r in A])),
            xi_by_n={int(n): float(fit([r for r in A if r["n"] == n])) for n in (4, 6, 8)})
print("A", json.dumps(resA, indent=1), flush=True)

# ------------------------------------------------------------ (B) thermal relaxation + depolarising floor
B = []; floor = 0.003; t2q, t1q = 4e-7, 4e-8
def qaoa(G, g, b):
    qc = QuantumCircuit(G.number_of_nodes()); qc.h(range(G.number_of_nodes()))
    for gg, bb in zip(g, b):
        for u, v in G.edges(): qc.cx(u, v); qc.rz(2 * gg, v); qc.cx(u, v)
        qc.rx(2 * bb, range(G.number_of_nodes()))
    return qc
for n in (8, 10):
    for gi in range(3):
        G = nx.random_regular_graph(3, n, seed=int(rng.integers(1e6))); m = G.number_of_edges()
        cuts = np.array([sum(((i >> u) & 1) != ((i >> v) & 1) for u, v in G.edges()) for i in range(2 ** n)])
        H = SparsePauliOp.from_sparse_list([("ZZ", [u, v], -0.5) for u, v in G.edges()], num_qubits=n) + SparsePauliOp("I" * n, coeffs=[m / 2])  # cut value operator
        for p in (1, 2):
            f = lambda x: -float(np.real(Statevector(qaoa(G, x[:p], x[p:])).expectation_value(H)))
            best = min((minimize(f, rng.uniform(0, np.pi, 2 * p), method="COBYLA", options=dict(maxiter=100)) for _ in range(3)), key=lambda r: r.fun)
            Eid = -best.fun; Emix = m / 2; qc = qaoa(G, best.x[:p], best.x[p:])
            for T in (25e-6, 50e-6, 100e-6, 200e-6):
                nm = NoiseModel()
                tr2 = thermal_relaxation_error(T, T, t2q).expand(thermal_relaxation_error(T, T, t2q))
                nm.add_all_qubit_quantum_error(tr2.compose(depolarizing_error(floor, 2)), ["cx"])
                nm.add_all_qubit_quantum_error(thermal_relaxation_error(T, T, t1q), ["rz", "rx", "h"])
                En, ops = noisy_energy(qc, H, nm, ["cx", "rz", "rx", "h"])
                g2 = ops.get("cx", 0); g1 = ops.get("rz", 0) + ops.get("rx", 0) + ops.get("h", 0)
                e2 = floor + t2q / T; e1 = t1q / T      # the simulator's parameterisation eps2 = floor + t2q/T2
                F = (1 - e2) ** g2 * (1 - e1) ** g1
                xi = 0.342 * (8.0 / min(max(n, 8), 12)) ** 0.82
                B.append(dict(n=n, T2=T, F=F, lost=lost(En, Eid, Emix), pred_cal=1 - F ** xi, pred_esp=1 - F))
    print("B n", n, "done", len(B), flush=True)
resB = dict(n_points=len(B), mae_calibrated=float(np.mean([abs(r["pred_cal"] - r["lost"]) for r in B])),
            mae_esp=float(np.mean([abs(r["pred_esp"] - r["lost"]) for r in B])),
            bias_calibrated=float(np.mean([r["pred_cal"] - r["lost"] for r in B])),
            bias_esp=float(np.mean([r["pred_esp"] - r["lost"] for r in B])),
            frac_cal_optimistic=float(np.mean([r["pred_cal"] < r["lost"] - 0.02 for r in B])),
            frac_esp_conservative=float(np.mean([r["pred_esp"] >= r["lost"] - 0.02 for r in B])))
print("B", json.dumps(resB, indent=1))
json.dump(dict(vqe=resA, thermal=resB, vqe_points=A, thermal_points=B), open("results/aer_extended.json", "w"), indent=1)
