"""Required comparison strategies for the Quantum Oracle Forensics challenge.

These baselines are deliberately simple and should not be mistaken for optimized
solvers. Each strategy uses only run(circuit, shots), just like the main solver.
"""
from collections import Counter
from math import log, log1p
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from solver import _bv_circuit, _grover_circuit, _simon_circuit, _classify_eight_bit, _decode_key

FAMILIES = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]


def _classical_query(n, m, x, run):
    qc = QuantumCircuit(n + m, m)
    for bit in range(n):
        if (x >> bit) & 1:
            qc.x(bit)
    qc.append(Gate("oracle", n + m, []), list(range(n + m)))
    for bit in range(m):
        qc.measure(n + bit, bit)
    counts = run(qc, 1)
    return _decode_key(max(counts, key=counts.get))


def random_guess(n, m, budget, families, run, rng=None):
    rng = rng or np.random.default_rng()
    choices = [f for f in FAMILIES if (m == 1) == (f not in ("simon", "simon2", "perm"))]
    family = str(rng.choice(choices))
    if family == "constant": secret = int(rng.integers(2))
    elif family == "linear": secret = [int(rng.integers(1, 256)), int(rng.integers(2))]
    elif family == "grover": secret = int(rng.integers(256))
    elif family == "simon": secret = int(rng.integers(1, 256))
    elif family == "simon2":
        a, b = map(int, rng.choice(np.arange(1, 256), 2, replace=False))
        secret = sorted([a, b, a ^ b])
    else: secret = None
    return family, secret, 0


def single_algorithm(n, m, budget, families, run, rng=None):
    """Fixed BV-only strategy for m=1; fixed Simon Fourier strategy for m=8."""
    if m == 1:
        shots = min(20, budget)
        counts = run(_bv_circuit(n), shots)
        hist = Counter({_decode_key(k): int(v) for k, v in counts.items()})
        if hist:
            a, count = hist.most_common(1)[0]
            if a != 0 and count >= max(3, int(0.4 * shots)):
                b = _classical_query(n, 1, 0, run) if budget > shots else 0
                return "linear", [int(a), int(b)], shots + int(budget > shots)
        return "constant", 0, shots
    shots = min(22, budget)
    counts = run(_simon_circuit(n), shots)
    samples = []
    for key, ct in counts.items():
        samples.extend([_decode_key(key) & 255] * int(ct))
    family, secret = _classify_eight_bit(samples)
    return family, secret, shots


def _simon_circuit(n):
    qc = QuantumCircuit(2 * n, n)
    for q in range(n): qc.h(q)
    qc.append(Gate("oracle", 2 * n, []), list(range(2 * n)))
    for q in range(n):
        qc.h(q)
        qc.measure(q, q)
    return qc


def classical_only(n, m, budget, families, run, rng=None):
    """No superposition: repeated basis queries plus distinct classical samples."""
    rng = rng or np.random.default_rng()
    used = 0
    if m == 8:
        order = rng.permutation(256)
        seen, differences = {}, []
        for x in order[:min(budget, 256)]:
            y = _classical_query(n, m, int(x), run); used += 1
            if y in seen:
                differences.append(int(x) ^ seen[y])
            else:
                seen[y] = int(x)
        if not differences:
            return "perm", None, used
        unique = set(differences)
        if len(unique) == 1:
            return "simon", int(next(iter(unique))), used
        span = {0}
        for v in unique:
            span |= {u ^ v for u in list(span)}
        if len(span) == 4:
            return "simon2", sorted(span - {0}), used
        return "perm", None, used

    # Recover a classical linear candidate from f(0) and f(e_i), using majority
    # reads to reduce readout/misfire errors. No quantum superposition is used.
    basis_values = {}
    if budget >= 27:
        for x in [0] + [1 << i for i in range(8)]:
            reads = [_classical_query(n, 1, x, run) for _ in range(3)]
            used += 3
            basis_values[x] = int(sum(reads) >= 2)
        b = basis_values[0]
        a = sum(((basis_values[1 << i] ^ b) << i) for i in range(8))
        if a != 0 and used + 9 <= budget:
            checks = [3, 7, 13]
            good = True
            for x in checks:
                reads = [_classical_query(n, 1, x, run) for _ in range(3)]
                used += 3
                if int(sum(reads) >= 2) != ((a & x).bit_count() % 2) ^ b:
                    good = False
            if good:
                return "linear", [int(a), int(b)], used
        if a == 0 and len(set(basis_values.values())) == 1:
            # Constant candidate; confirm at distinct inputs if budget allows.
            vals = list(basis_values.values())
            if all(v == vals[0] for v in vals):
                return "constant", int(vals[0]), used

    # Spend the remainder on random distinct inputs. Reserve up to two queries
    # to recheck a candidate positive if exactly one is observed.
    already = set(basis_values)
    pool = [int(x) for x in rng.permutation(256) if int(x) not in already]
    observed = {x: y for x, y in basis_values.items()}
    target_queries = min(budget, used + len(pool))
    while used < target_queries and pool:
        x = pool.pop()
        y = _classical_query(n, 1, x, run); used += 1
        observed[x] = y
    positives = [x for x, y in observed.items() if y == 1]
    if len(positives) == 1 and used + 2 <= budget:
        t = positives[0]
        checks = [_classical_query(n, 1, t, run) for _ in range(2)]; used += 2
        if sum(checks) == 2:
            return "grover", int(t), used
    # Use only distinct-input observations for density; the basis majority values
    # are included once each, not once per repeat.
    k, total = sum(observed.values()), max(1, len(observed))
    if k / total > 0.78: return "constant", 1, used
    if k / total < 0.08: return "constant", 0, used
    ll_bal = k * log(0.4808) + (total-k) * log1p(-0.4808)
    ll_rand = k * log(0.29648) + (total-k) * log1p(-0.29648)
    return ("balanced", None, used) if ll_bal >= ll_rand else ("randfun", None, used)


def nonadaptive_fixed_plan(n, m, budget, families, run, rng=None):
    """Fixed experiment schedule selected before any outcomes are observed."""
    rng = rng or np.random.default_rng()
    if m == 8:
        shots = min(22, budget)
        counts = run(_simon_circuit(n), shots)
        ys = []
        for key, ct in counts.items(): ys.extend([_decode_key(key) & 255] * int(ct))
        family, secret = _classify_eight_bit(ys)
        return family, secret, shots

    # Fixed budget split: BV first, Grover circuit second, classical sample last.
    bv_shots = min(8, max(1, budget // 8))
    remaining = budget - bv_shots
    grover_shots = min(3, remaining // 7)
    grover_cost = 7 * grover_shots
    classical_count = max(0, remaining - grover_cost)
    bv_counts = run(_bv_circuit(n), bv_shots)
    grover_counts = {}
    if grover_shots:
        grover_counts = run(_grover_circuit(n, 7), grover_shots)
    classical = {}
    for x in range(min(classical_count, 256)):
        classical[x] = _classical_query(n, 1, x, run)
    used = bv_shots + grover_cost + len(classical)
    hist = Counter({_decode_key(k): int(v) for k, v in bv_counts.items()})
    if hist:
        a, ct = hist.most_common(1)[0]
        if a and ct >= max(3, int(0.45 * bv_shots)):
            b = classical.get(0, 0)
            return "linear", [int(a), int(b)], used
    k, total = sum(classical.values()), max(1, len(classical))
    if k / total > 0.78: return "constant", 1, used
    if k / total > 0.16:
        ll_bal = k * log(0.4808) + (total-k) * log1p(-0.4808)
        ll_rand = k * log(0.29648) + (total-k) * log1p(-0.29648)
        return ("balanced", None, used) if ll_bal >= ll_rand else ("randfun", None, used)
    if grover_counts:
        t = _decode_key(max(grover_counts, key=grover_counts.get))
        if classical.get(t, 0) == 1:
            return "grover", int(t), used
    return "constant", 0, used
