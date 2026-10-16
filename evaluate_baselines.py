"""Compare adaptive solver with required baselines on local noisy oracles."""
import argparse
import json
import numpy as np
from solver import solve
from simulator import LocalOracle, local_run
from baselines import FAMILIES, random_guess, classical_only, single_algorithm, nonadaptive_fixed_plan

SECRETLESS = {"balanced", "randfun", "perm"}
STRATEGIES = {
    "adaptive": solve,
    "classical_only": classical_only,
    "single_algorithm": single_algorithm,
    "nonadaptive": nonadaptive_fixed_plan,
    "random_guess": random_guess,
}


def _one_trial(family, strategy_name, budget, p_misfire, p_readout, seed):
    rng = np.random.default_rng(seed)
    m = 8 if family in ("simon", "simon2", "perm") else 1
    oracle = LocalOracle(family, m=m, seed=int(rng.integers(2**31)))
    raw_run = local_run(oracle, p_misfire=p_misfire, p_readout=p_readout,
                        seed=int(rng.integers(2**31)))
    usage = {"queries": 0}
    def run(circuit, shots):
        cost = sum(1 for inst in circuit.data if inst.operation.name == "oracle") * int(shots)
        usage["queries"] += cost
        if usage["queries"] > budget:
            raise RuntimeError(f"query budget exceeded: {usage['queries']} > {budget}")
        return raw_run(circuit, shots)
    fn = STRATEGIES[strategy_name]
    if strategy_name == "random_guess":
        predicted, secret, reported_used = fn(8, m, budget, FAMILIES, run, rng)
    else:
        predicted, secret, reported_used = fn(8, m, budget, FAMILIES, run, rng) if strategy_name != "adaptive" else fn(8, m, budget, FAMILIES, run)
    # Actual wrapper count is authoritative; compare against baseline's reported count.
    family_ok = predicted == family
    secret_ok = family in SECRETLESS or secret == oracle.secret
    return {"true": family, "predicted": predicted, "secret_true": oracle.secret,
            "secret_predicted": secret, "queries_used": usage["queries"],
            "reported_queries": reported_used, "correct": bool(family_ok and secret_ok)}


def evaluate_baselines(repeats=3, budget=60, p_misfire=0.04, p_readout=0.02, seed=19):
    rng = np.random.default_rng(seed)
    summary = {}
    for strategy in STRATEGIES:
        trials = []
        for family in FAMILIES:
            for _ in range(repeats):
                trials.append(_one_trial(family, strategy, budget, p_misfire, p_readout,
                                         int(rng.integers(2**31))))
        per_family = {}
        for family in FAMILIES:
            rows = [r for r in trials if r["true"] == family]
            per_family[family] = sum(r["correct"] for r in rows) / max(1, len(rows))
        correct = [r for r in trials if r["correct"]]
        summary[strategy] = {
            "accuracy": sum(r["correct"] for r in trials) / max(1, len(trials)),
            "worst_family_accuracy": min(per_family.values()),
            "per_family_accuracy": per_family,
            "mean_queries": sum(r["queries_used"] for r in trials) / max(1, len(trials)),
            "query_saving_on_correct_tasks": (sum(1 - r["queries_used"] / budget for r in correct) / len(correct) if correct else None),
            "details": trials,
        }
    return {"seed": seed, "budget": budget,
            "noise": {"misfire": p_misfire, "readout": p_readout}, "strategies": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--misfire", type=float, default=0.04)
    parser.add_argument("--readout", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    print(json.dumps(evaluate_baselines(args.repeats, args.budget, args.misfire,
                                        args.readout, args.seed), indent=2))
