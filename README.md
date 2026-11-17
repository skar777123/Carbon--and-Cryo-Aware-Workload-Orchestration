# EcoQ simulator and experiments
Python 3.13, numpy, scipy, pandas, matplotlib, networkx, torch (CPU), qiskit 2.6.2, qiskit-aer 0.18.2.
- `env.py` simulator (all parameters = assumptions, see DEFAULTS/QPUS)  - `policies.py` baselines + EcoQ planner
- `ppo.py` masked PPO (+behaviour-cloning init, KL anchor) - `train_all.py` trains 3 BC-PPO + 2 scratch-PPO models
- `validate_aer.py`, `calib_xi.py` fidelity-model validation - `run_experiments.py {main_heur,main_rl,ablate,sens,pareto}`
- `analyze.py` writes ../paper/tables, figs, numbers.tex from `results/*.csv|json`.
Order: validate_aer -> calib_xi -> train_all -> run_experiments (5 modes) -> analyze -> pdflatex main.tex (twice).
Disclosure: results/ are from the re-run after fixing a carbon-forecast defect (forecast error collapsed at long lead). Earlier buggy outputs are not included.
Carbon traces are the author-supplied 2025 hourly profiles in data/traces_2025.csv (five zones, labelled empirically calibrated profiles, not raw grid measurements).

## Update: IBM hardware validation and fidelity-model default
`results/ibm_hardware.json` holds 9 QAOA jobs run on `ibm_fez` via `ibm_hardware_validation.py` (job IDs included). Plain ESP (xi=1)
matched the measurements (MAE 0.027); the Aer-fitted exponent (xi<1) was optimistic (MAE 0.12), so the simulator default is now
`xi_mode="esp"` and all experiments were rerun. Earlier Aer-calibrated results are kept in `results_xical/` for reference.
Scheduling experiments remain simulated; only the fidelity model was validated on hardware.
