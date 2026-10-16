"""Reproducible local evaluation, confusion matrix, budget sweeps, and stress tests."""
from collections import defaultdict
import argparse
import json
import numpy as np
from solver import solve
from simulator import LocalOracle, local_run

FAMILIES = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]
SECRETLESS = {"balanced", "randfun", "perm"}


def _run_trial(family, budget, p_misfire, p_readout, seed):
    rng = np.random.default_rng(seed)
    m = 8 if family in ("simon", "simon2", "perm") else 1
    oracle = LocalOracle(family, m=m, seed=int(rng.integers(2**31)))
    raw_run = local_run(oracle, p_misfire=p_misfire, p_readout=p_readout,
                        seed=int(rng.integers(2**31)))
    usage = {"queries": 0}
    def counted_run(circuit, shots):
        cost = sum(1 for inst in circuit.data if inst.operation.name == "oracle") * int(shots)
        usage["queries"] += cost
        if usage["queries"] > budget:
            raise RuntimeError(f"solver exceeded query budget: {usage['queries']} > {budget}")
        return raw_run(circuit, shots)
    predicted, secret = solve(8, m, budget, FAMILIES, counted_run)
    family_ok = predicted == family
    secret_ok = family in SECRETLESS or secret == oracle.secret
    return {"true": family, "predicted": predicted, "secret_true": oracle.secret,
            "secret_predicted": secret, "queries_used": usage["queries"],
            "query_saving": 1 - usage["queries"] / budget,
            "correct": bool(family_ok and secret_ok)}


def evaluate(repeats=3, budget=60, p_misfire=0.04, p_readout=0.02, seed=7):
    rng = np.random.default_rng(seed)
    trials = []
    for family in FAMILIES:
        for _ in range(repeats):
            trials.append(_run_trial(family, budget, p_misfire, p_readout,
                                     int(rng.integers(2**31))))
    confusion = {t: {p: 0 for p in FAMILIES} for t in FAMILIES}
    per_family = {}
    for trial in trials:
        confusion[trial["true"]][trial["predicted"]] += 1
    for family in FAMILIES:
        rows = [t for t in trials if t["true"] == family]
        per_family[family] = sum(t["correct"] for t in rows) / max(1, len(rows))
    correct_rows = [t for t in trials if t["correct"]]
    query_saving = (sum(t["query_saving"] for t in correct_rows) / len(correct_rows)
                    if correct_rows else None)
    return {"seed": seed, "budget": budget, "noise": {"misfire": p_misfire, "readout": p_readout},
            "trials": len(trials), "accuracy": sum(t["correct"] for t in trials) / max(1, len(trials)),
            "worst_family_accuracy": min(per_family.values()),
            "query_saving_on_correct_tasks": query_saving,
            "per_family_accuracy": per_family, "confusion_matrix": confusion, "details": trials}


def budget_sweep(repeats=2, budgets=(20, 30, 40, 50, 60), seed=7, p_misfire=0.04, p_readout=0.02):
    return {str(b): evaluate(repeats, b, p_misfire, p_readout, seed + b) for b in budgets}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--misfire", type=float, default=0.04)
    parser.add_argument("--readout", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    if args.sweep:
        result = budget_sweep(args.repeats, seed=args.seed, p_misfire=args.misfire, p_readout=args.readout)
    else:
        result = evaluate(args.repeats, args.budget, args.misfire, args.readout, args.seed)
    print(json.dumps(result, indent=2))
