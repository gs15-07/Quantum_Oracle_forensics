"""Quantum Oracle Forensics solver for the QryptoNox / Qiskit Fall Fest task.

Public API required by the judge:
    solve(n, m, budget, families, run) -> (family, secret)

Only ``run`` is used to query the sealed oracle. All candidate generation,
statistics, and experiment selection are local calculations.
"""
from collections import Counter
from itertools import combinations
from math import log, log1p, exp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

P_MISFIRE = 0.04
P_READOUT = 0.02


def _append_oracle(qc, n, m):
    qc.append(Gate("oracle", n + m, []), list(range(n + m)))


def _decode_key(key):
    return int(key.replace(" ", ""), 2) if key.replace(" ", "") else 0


def _classical_query(n, m, x, run):
    """One classical input query, output register initialized to |0...0>."""
    qc = QuantumCircuit(n + m, m)
    for bit in range(n):
        if (x >> bit) & 1:
            qc.x(bit)
    _append_oracle(qc, n, m)
    for bit in range(m):
        qc.measure(n + bit, bit)
    counts = run(qc, 1)
    key = max(counts, key=counts.get)
    return _decode_key(key)


def _bv_circuit(n):
    """Bernstein-Vazirani circuit for a one-bit function using phase kickback."""
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(n)
    for q in range(n):
        qc.h(q)
    _append_oracle(qc, n, 1)
    for q in range(n):
        qc.h(q)
        qc.measure(q, q)
    return qc


def _simon_circuit(n):
    """Simon Fourier sample. Output is unmeasured; only input is measured."""
    qc = QuantumCircuit(2 * n, n)
    for q in range(n):
        qc.h(q)
    _append_oracle(qc, n, n)
    for q in range(n):
        qc.h(q)
        qc.measure(q, q)
    return qc


def _grover_circuit(n, iterations):
    """Grover search over f(x)=1, with phase kickback on the output qubit."""
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(n)  # |-> target: oracle output becomes a phase oracle
    for q in range(n):
        qc.h(q)
    for _ in range(iterations):
        _append_oracle(qc, n, 1)
        # Diffusion: H^n X^n (multi-controlled-Z) X^n H^n.
        for q in range(n):
            qc.h(q)
        for q in range(n):
            qc.x(q)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q in range(n):
            qc.x(q)
        for q in range(n):
            qc.h(q)
    for q in range(n):
        qc.measure(q, q)
    return qc


def _majority(values):
    return int(sum(int(v) for v in values) * 2 >= len(values)) if values else 0


def _parity(x):
    return x.bit_count() & 1


def _log_binom_likelihood(k, n, p):
    # Binomial likelihood; constants common to hypotheses cancel in comparisons.
    if p <= 0.0:
        return 0.0 if k == 0 else -1e300
    if p >= 1.0:
        return 0.0 if k == n else -1e300
    return k * log(p) + (n - k) * log1p(-p)


def _logsumexp(values):
    values = list(values)
    if not values:
        return -1e300
    m = max(values)
    if m < -1e299:
        return m
    return m + log(sum(exp(v - m) for v in values))


def _syndrome_noise_probs(u, v=None, p=P_READOUT):
    """Distribution of H^T e for independent readout-error bits e_i~Bernoulli(p).

    Returns P(parity(e&u)=s) for Simon (v=None), or the four joint syndrome
    probabilities for a two-generator Simon2 subspace.
    """
    size = 2 if v is None else 4
    probs = [0.0] * size
    probs[0] = 1.0
    for bit in range(8):
        toggle = (u >> bit) & 1
        if v is not None:
            toggle |= ((v >> bit) & 1) << 1
        nxt = [0.0] * size
        for s, pr in enumerate(probs):
            nxt[s] += pr * (1 - p)
            nxt[s ^ toggle] += pr * p
        probs = nxt
    return probs


def _simon2_subspaces(n=8):
    """Enumerate each 2D GF(2) subspace once as its sorted nonzero triple."""
    found = set()
    for a in range(1, 1 << n):
        for b in range(a + 1, 1 << n):
            triple = tuple(sorted((a, b, a ^ b)))
            if len(set(triple)) == 3:
                found.add(triple)
    return sorted(found)


def _classify_eight_bit(fourier_samples, p_misfire=P_MISFIRE, p_readout=P_READOUT):
    """Bayesian family evidence and MAP secret for simon/simon2/perm.

    Readout noise is modelled exactly through the syndrome distribution. A
    misfired oracle leaves the input uniform, contributing 1/256 to every
    measured value. Evidence averages over the uniform secret prior, avoiding
    the multiple-candidate bias of comparing maximum likelihood alone.
    """
    ys = [int(y) & 255 for y in fourier_samples]
    if len(ys) < 5:
        return "perm", None
    nobs = len(ys)
    perm_log = -nobs * log(256)

    simon_scores = {}
    for s in range(1, 256):
        synd_probs = _syndrome_noise_probs(s, p=p_readout)
        lp = 0.0
        for y in ys:
            syndrome = _parity(y & s)
            likelihood = p_misfire / 256 + (1 - p_misfire) * synd_probs[syndrome] / 128
            lp += log(max(likelihood, 1e-300))
        simon_scores[s] = lp
    simon_evidence = _logsumexp(simon_scores.values()) - log(len(simon_scores))
    best_s = max(simon_scores, key=simon_scores.get)

    subspace_scores = {}
    for triple in _simon2_subspaces():
        u, v = triple[0], triple[1]
        syndrome_probs = _syndrome_noise_probs(u, v, p_readout)
        lp = 0.0
        for y in ys:
            syndrome = _parity(y & u) | (_parity(y & v) << 1)
            likelihood = p_misfire / 256 + (1 - p_misfire) * syndrome_probs[syndrome] / 64
            lp += log(max(likelihood, 1e-300))
        subspace_scores[triple] = lp
    simon2_evidence = _logsumexp(subspace_scores.values()) - log(len(subspace_scores))
    best_triple = max(subspace_scores, key=subspace_scores.get)

    # Equal family priors; likelihood evidence is marginalized over secrets.
    evidence = {"simon": simon_evidence, "simon2": simon2_evidence, "perm": perm_log}
    family = max(evidence, key=evidence.get)
    if family == "simon":
        return family, int(best_s)
    if family == "simon2":
        return family, list(best_triple)
    return "perm", None


def _solve_one_bit(n, budget, run):
    remaining = int(budget)

    def spend(cost):
        nonlocal remaining
        if cost > remaining:
            raise RuntimeError("query budget exhausted")
        remaining -= cost

    # Classical fingerprint: repeated x=0 helps estimate the offset / constant.
    zero_reads = [_classical_query(n, 1, 0, run) for _ in range(3)]
    spend(3)
    sample_xs = list(range(1, 6))
    samples = {0: _majority(zero_reads)}
    for x in sample_xs:
        samples[x] = _classical_query(n, 1, x, run)
        spend(1)

    # BV has a concentrated peak at a for linear functions; constants peak at 0.
    # Keep a small reserve for verification at low budgets.
    bv_shots = min(10, max(1, remaining - 6))
    bv_counts = run(_bv_circuit(n), bv_shots)
    spend(bv_shots)
    bv_hist = Counter({_decode_key(k): int(v) for k, v in bv_counts.items()})
    if bv_hist:
        candidate_a, candidate_count = bv_hist.most_common(1)[0]
        if (candidate_a != 0 and candidate_count >= max(4, int(0.45 * bv_shots))
                and remaining >= 6):
            # Independently verify parity relation and infer b from f(0).
            # Two independent inputs are enough for a quick check; remaining
            # budget is retained for the other family branches.
            verify_xs = [0] + [1 << bit for bit in range(n) if (candidate_a >> bit) & 1][:1]
            verify = {}
            for x in verify_xs:
                reads = [_classical_query(n, 1, x, run) for _ in range(3)]
                spend(3)
                verify[x] = _majority(reads)
            b = verify[0]
            if all(y == (_parity(candidate_a & x) ^ b) for x, y in verify.items()):
                return "linear", [int(candidate_a), int(b)]

    # A near-all-one fingerprint is consistent with constant-one. Use four
    # fresh points to avoid confusing a lucky dense-function sample for a constant.
    if sum(samples.values()) >= 5 and remaining >= 4:
        extra = []
        for x in range(6, 10):
            extra.append(_classical_query(n, 1, x, run))
            spend(1)
        if sum(samples.values()) + sum(extra) >= 9:
            return "constant", 1

    # If observations already look dense, spend the remaining budget on samples
    # at distinct inputs; this is enough to separate fixed-balanced from p=.3 random.
    ones = sum(samples.values())
    if ones >= 2:
        while remaining > 0 and len(samples) < min(50, 256):
            x = next(v for v in range(256) if v not in samples)
            samples[x] = _classical_query(n, 1, x, run)
            spend(1)
        k, total = sum(samples.values()), len(samples)
        # Effective observed-one probabilities under the specified noise:
        # balanced: .5*.9416 + .5*.02 = .4808; randfun: .3*.9416 + .7*.02=.29648.
        p_balanced, p_randfun = 0.4808, 0.29648
        ll_bal = _log_binom_likelihood(k, total, p_balanced)
        ll_rand = _log_binom_likelihood(k, total, p_randfun)
        return ("balanced", None) if ll_bal >= ll_rand else ("randfun", None)

    # A single positive among the first few points can occur by chance for
    # randfun. Expand the fingerprint to 18 distinct inputs before treating the
    # oracle as sparse; dense families almost always reveal themselves by then.
    sparse_probe_target = min(18, len(samples) + max(0, remaining - 8))
    while sum(samples.values()) < 2 and len(samples) < sparse_probe_target and remaining > 0:
        x = next(v for v in range(256) if v not in samples)
        samples[x] = _classical_query(n, 1, x, run)
        spend(1)
    if sum(samples.values()) >= 2:
        while remaining > 0 and len(samples) < 50:
            x = next(v for v in range(256) if v not in samples)
            samples[x] = _classical_query(n, 1, x, run)
            spend(1)
        k, total = sum(samples.values()), len(samples)
        ll_bal = _log_binom_likelihood(k, total, 0.4808)
        ll_rand = _log_binom_likelihood(k, total, 0.29648)
        return ("balanced", None) if ll_bal >= ll_rand else ("randfun", None)

    # With very few ones, distinguish sparse Grover from constant-zero using
    # amplitude amplification. Seven iterations with three shots leave room for
    # candidate checks under the full 60-query budget.
    candidate_counts = Counter()
    for x, y in samples.items():
        if y == 1:
            candidate_counts[x] += 1
    if remaining >= 4:
        iterations = min(7, max(1, (remaining - 8) // 3))
        shots = min(3, remaining // iterations)
        if shots > 0:
            grover_counts = run(_grover_circuit(n, iterations), shots)
            spend(iterations * shots)
            for key, ct in grover_counts.items():
                candidate_counts[_decode_key(key)] += int(ct)

    # Triage up to three candidates with one query each. Only candidates that
    # return 1 receive two additional verification queries; this preserves budget
    # while preventing tied Grover measurement counts from hiding the true target.
    for x, _ in candidate_counts.most_common(3):
        if remaining < 1:
            break
        first = _classical_query(n, 1, x, run)
        spend(1)
        if first != 1 or remaining < 2:
            continue
        follow = [_classical_query(n, 1, x, run) for _ in range(2)]
        spend(2)
        if first + sum(follow) >= 2:
            # Confirm the sparse signature on fresh, non-candidate inputs too.
            checks = []
            for probe in range(240, 245):
                if remaining <= 0:
                    break
                if probe == x or probe in samples:
                    continue
                checks.append(_classical_query(n, 1, probe, run))
                spend(1)
            other_ones = sum(y for xx, y in samples.items() if xx != x)
            if other_ones == 0 and sum(checks) == 0:
                return "grover", int(x)

    # If we have remaining queries, use distinct classical inputs to distinguish
    # a rare random-function sample from constant-zero. A single positive is not
    # enough to claim Grover without successful repeated verification.
    while remaining > 0 and len(samples) < 256:
        x = next(v for v in range(256) if v not in samples)
        samples[x] = _classical_query(n, 1, x, run)
        spend(1)
        if sum(samples.values()) >= 2:
            k, total = sum(samples.values()), len(samples)
            ll_bal = _log_binom_likelihood(k, total, 0.4808)
            ll_rand = _log_binom_likelihood(k, total, 0.29648)
            # If many positives emerge, this is not a single marked item.
            return ("balanced", None) if ll_bal >= ll_rand else ("randfun", None)
    return "constant", 0


def solve(n, m, budget, families, run):
    """Classify a sealed oracle and recover its secret under the query budget.

    ``run(circuit, shots)`` is the judge-supplied Qiskit 2.x runner. The solver
    does not inspect oracle internals; every decision is based on circuit results.
    """
    if n != 8 or m not in (1, 8):
        raise ValueError("This challenge specifies n=8 and m in {1,8}.")
    if budget < 1:
        raise ValueError("budget must be positive")
    if m == 1:
        return _solve_one_bit(n, budget, run)

    # Simon Fourier sampling: each shot costs one oracle gate.
    shots = min(22, int(budget))
    counts = run(_simon_circuit(n), shots)
    ys = []
    for key, count in counts.items():
        y = _decode_key(key) & 255
        ys.extend([y] * int(count))
    return _classify_eight_bit(ys)
