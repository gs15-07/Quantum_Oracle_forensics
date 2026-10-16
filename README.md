# Quantum Oracle Forensics — solver, simulator, evaluation harness

A development implementation for the QryptoNox / IBM Qiskit Fall Fest 2026
challenge. It is organized around three components:

- `solver.py`: judge-facing `solve(n, m, budget, families, run)` implementation.
- `simulator.py`: synthetic oracle generators and local noisy statevector runner.
- `evaluate.py`: randomized evaluation, confusion matrix, per-family accuracy,
  worst-family accuracy, query savings, and budget/noise sweeps.
- `baselines.py`: classical-only, single-algorithm, random-guessing, and fixed-plan
  non-adaptive baselines.
- `evaluate_baselines.py`: compares the adaptive solver with the baselines.
- `test_oracles.py`: public reference-instance checks from the challenge PDF.

## Install and test

Python 3.10+ is required.

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_oracles.py
python evaluate.py --repeats 5 --budget 60 --seed 7
python evaluate.py --repeats 3 --budget 60 --misfire 0.08 --readout 0.04
python evaluate.py --repeats 2 --sweep
python evaluate_baselines.py --repeats 3 --budget 60 --seed 19
```

The solver uses placeholder `Gate("oracle", n+m, [])` instructions and makes all
queries through the supplied `run` callable. It never reads local oracle tables.
The local simulator is only for development and evaluation.

## Algorithm overview

- One-bit output: classical fingerprints, Bernstein–Vazirani with phase kickback,
  classical secret verification, Grover search for a sparse marked input, and
  noise-adjusted density likelihoods for balanced vs random functions.
- Eight-bit output: Simon Fourier samples; compare marginal likelihood under
  Simon, Simon2, and permutation hypotheses while modelling oracle misfires and
  readout bit flips. Simon2 candidates are 2D subspaces over GF(2).
- Evaluation: family confusion matrix, exact-secret correctness, query usage and
  savings, worst-family accuracy, noise stress test, budget sweep, and comparison
  against classical-only, fixed-algorithm, random-guessing, and non-adaptive baselines.

## Important limitations / submission checklist

This is a substantial baseline, not a guarantee of winning or a claim of measured
accuracy. The environment used to author this bundle did not have Qiskit installed,
so the Qiskit-backed circuit execution has not been runtime-verified here. Install
the declared requirements, run the reference tests, and then evaluate many seeds.

Before submitting:
1. Confirm the judge's Qiskit 2.x version accepts the placeholder gate exactly as
   specified and preserves the expected classical-bit string ordering.
2. Check query cost at every branch and enforce `oracle gates × shots <= budget`.
3. Tune thresholds using independent training seeds; do not tune on hidden tests.
4. Add explicit query-count logging, ablations, and classical/single-algorithm/random
   baselines to the report.
5. Report failure modes honestly, especially low-budget secret recovery and noisy
   Simon2 discrimination.

## AI-use disclosure

AI assistance was used to draft and review parts of the code and documentation.
The team must independently execute, inspect, validate, and disclose the final
implementation according to the event's rules.
