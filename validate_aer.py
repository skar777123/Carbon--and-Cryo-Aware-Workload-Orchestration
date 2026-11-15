"""Validate the estimated-success-probability (ESP) fidelity model against Qiskit Aer noisy simulation.

For random 3-regular Max-Cut graphs we run QAOA (p=1,2) with optimised angles in an ideal statevector
simulation, then re-evaluate the *same* circuit under a depolarising + readout noise model for a sweep of
two-qubit error rates.  The ESP-model prediction of the noisy energy is
        <C>_noisy  ~=  F * <C>_ideal + (1-F) * <C>_mixed,       F = prod(1-e_g),
(<C>_mixed = m/2 for Max-Cut) and is compared with the Aer value (single-/two-qubit depolarising error
and readout error applied to the same transpiled circuit).  Output: results/aer_validation.json
"""
import json, sys, time, itertools
import numpy as np, networkx as nx
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

rng = np.random.default_rng(2026)


def qaoa_circuit(G, gammas, betas):
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for u, v in G.edges():
            qc.cx(u, v); qc.rz(2 * g, v); qc.cx(u, v)       # exp(-i g Z_u Z_v)
        qc.rx(2 * b, range(n))
    return qc


def cut_value(bits, edges):
    return sum(1 for u, v in edges if bits[u] != bits[v])


def ideal_energy(G, x, p):
    qc = qaoa_circuit(G, x[:p], x[p:])
    sv = Statevector(qc)
    pr = sv.probabilities()
    n = G.number_of_nodes()
    edges = list(G.edges())
    cuts = np.array([cut_value([(i >> q) & 1 for q in range(n)], edges) for i in range(2 ** n)])
    return float(pr @ cuts), cuts


def main():
    results = []
    t0 = time.time()
    jobs_ = [(n_, gi_) for n_ in (8, 10, 12) for gi_ in range(3)]
    for (n, gi) in jobs_:
        G = nx.random_regular_graph(3, n, seed=int(rng.integers(1e6)))
        m = G.number_of_edges()
        for p in (1, 2):
            best = None
            for trial in range(4):
                x0 = rng.uniform(0, np.pi, 2 * p)
                r = minimize(lambda x: -ideal_energy(G, x, p)[0], x0, method="COBYLA", options=dict(maxiter=120))
                if best is None or r.fun < best.fun:
                    best = r
            x = best.x
            E_ideal, cuts = ideal_energy(G, x, p)
            maxcut = cuts.max()
            qc = qaoa_circuit(G, x[:p], x[p:])
            qcm = qc.copy(); qcm.measure_all()
            for e2 in (0.0, 0.005, 0.01, 0.02, 0.04):
                e1, ero = e2 / 10, e2 / 2
                nm = NoiseModel()
                if e2 > 0:
                    nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["cx"])
                    nm.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ["rz", "rx", "h"])
                    nm.add_all_qubit_readout_error(ReadoutError([[1 - ero, ero], [ero, 1 - ero]]))
                sim = AerSimulator(noise_model=nm, basis_gates=["cx", "rz", "rx", "h"])
                tq = transpile(qcm, sim, basis_gates=["cx", "rz", "rx", "h"], optimization_level=0)
                ops = tq.count_ops()
                g2, g1 = ops.get("cx", 0), ops.get("rz", 0) + ops.get("rx", 0) + ops.get("h", 0)
                shots = 10000
                counts = sim.run(tq, shots=shots, seed_simulator=int(rng.integers(1e6))).result().get_counts()
                tot = 0.0
                for bs, c in counts.items():
                    bits = [int(ch) for ch in bs[::-1]]
                    tot += c * cut_value(bits, G.edges())
                E_aer = tot / shots
                # depolarising error of strength e on k qubits keeps the state with prob (1-e) -> ESP
                F = (1 - e2) ** g2 * (1 - e1) ** g1 * (1 - ero) ** n
                E_esp = F * E_ideal + (1 - F) * (m / 2)
                results.append(dict(n=n, graph=gi, p=p, e2=e2, g2=g2, g1=g1, F=F, E_ideal=E_ideal, E_aer=E_aer,
                                    E_esp=E_esp, maxcut=float(maxcut), m=m,
                                    ar_aer=E_aer / maxcut, ar_esp=E_esp / maxcut, ar_ideal=E_ideal / maxcut))
        print("n", n, "graph", gi, "done", round(time.time() - t0, 1), "s", flush=True)
    a = np.array([r["E_aer"] for r in results]); b = np.array([r["E_esp"] for r in results])
    ia = np.array([r["ar_aer"] for r in results]); ib = np.array([r["ar_esp"] for r in results])
    Fs = np.array([r["F"] for r in results])
    # bias in "fraction of ideal-to-mixed gap lost":  1 - (E_noisy - E_mixed)/(E_ideal - E_mixed)
    gaps = np.array([r["E_ideal"] - r["m"] / 2 for r in results])
    lost_aer = 1 - (a - np.array([r["m"] / 2 for r in results])) / gaps
    lost_esp = 1 - Fs
    summ = dict(n_qubits=[8, 10, 12], n_graphs=3, n_points=len(results),
                mae_energy=float(np.mean(np.abs(a - b))),
                mae_approx_ratio=float(np.mean(np.abs(ia - ib))),
                max_abs_err_ar=float(np.max(np.abs(ia - ib))),
                pearson_r_lost=float(np.corrcoef(lost_aer, lost_esp)[0, 1]),
                mae_lost_fraction=float(np.mean(np.abs(lost_aer - lost_esp))),
                mean_signed_err_lost=float(np.mean(lost_esp - lost_aer)),
                shots=10000)
    json.dump(dict(summary=summ, points=results), open("results/aer_validation.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
