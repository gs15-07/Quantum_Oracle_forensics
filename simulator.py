"""Local noisy-oracle simulator for development and evaluation.

The simulator implements the Qiskit circuit subset used by solver.py and the
challenge noise model: each oracle gate independently misfires with probability
p_misfire, and each actually measured bit independently flips with probability
p_readout. It is a local test harness.
"""
from collections import Counter
import numpy as np


class LocalOracle:
    def __init__(self, family, n=8, m=1, seed=None, **params):
        self.family, self.n, self.m = family, n, m
        self.rng = np.random.default_rng(seed)
        self.params = params
        self.secret = None
        self.table = self._make_table()

    def _make_table(self):
        n, m, family = self.n, self.m, self.family
        N = 1 << n
        rng = self.rng
        if family == "constant":
            c = int(self.params.get("c", rng.integers(2)))
            self.secret = c
            return np.full(N, c, dtype=np.uint16)
        if family == "balanced":
            table = np.zeros(N, dtype=np.uint16)
            table[rng.choice(N, N // 2, replace=False)] = 1
            return table
        if family == "linear":
            a = int(self.params.get("a", rng.integers(1, N)))
            b = int(self.params.get("b", rng.integers(2)))
            if a == 0:
                raise ValueError("linear family requires a != 0")
            self.secret = [a, b]
            return np.array([((a & x).bit_count() & 1) ^ b for x in range(N)], dtype=np.uint16)
        if family == "grover":
            t = int(self.params.get("t", rng.integers(N)))
            self.secret = t
            table = np.zeros(N, dtype=np.uint16)
            table[t] = 1
            return table
        if family == "randfun":
            return (rng.random(N) < 0.3).astype(np.uint16)
        if family == "simon":
            s = int(self.params.get("s", rng.integers(1, N)))
            if not 1 <= s < N:
                raise ValueError("Simon secret must be in 1..255")
            self.secret = s
            return np.array([min(x, x ^ s) for x in range(N)], dtype=np.uint16)
        if family == "simon2":
            H = self.params.get("H")
            if H is None:
                while True:
                    a, b = map(int, rng.choice(np.arange(1, N), 2, replace=False))
                    if a != b:
                        H = {0, a, b, a ^ b}
                        break
            H = set(map(int, H))
            if len(H) != 4 or 0 not in H or any((a ^ b) not in H for a in H for b in H):
                raise ValueError("H must be a 2D GF(2) subspace {0,a,b,a xor b}")
            self.secret = sorted(H - {0})
            return np.array([min(x ^ h for h in H) for x in range(N)], dtype=np.uint16)
        if family == "perm":
            if self.params.get("affine_example", False):
                return np.array([(37 * x + 11) % 256 for x in range(N)], dtype=np.uint16)
            return rng.permutation(N).astype(np.uint16)
        raise ValueError(f"unknown family: {family}")


def _apply_single(state, q, mat):
    stride = 1 << q
    for base in range(0, len(state), stride << 1):
        for off in range(stride):
            i, j = base + off, base + off + stride
            a, b = state[i], state[j]
            state[i] = mat[0, 0] * a + mat[0, 1] * b
            state[j] = mat[1, 0] * a + mat[1, 1] * b


def _apply_cx(state, control, target):
    for i in range(len(state)):
        if ((i >> control) & 1) and not ((i >> target) & 1):
            j = i | (1 << target)
            state[i], state[j] = state[j], state[i]


def _apply_mcx(state, controls, target):
    control_mask = sum(1 << q for q in controls)
    target_mask = 1 << target
    for i in range(len(state)):
        if (i & control_mask) == control_mask and not (i & target_mask):
            j = i | target_mask
            state[i], state[j] = state[j], state[i]


def local_run(oracle, p_misfire=0.04, p_readout=0.02, seed=None):
    """Return a Qiskit-compatible ``run(circuit, shots)`` function."""
    rng = np.random.default_rng(seed)

    def run(circuit, shots):
        nq, nc = circuit.num_qubits, circuit.num_clbits
        results = Counter()
        for _ in range(int(shots)):
            state = np.zeros(1 << nq, dtype=np.complex128)
            state[0] = 1.0
            measured = {}
            for inst in circuit.data:
                op = inst.operation
                qs = [circuit.find_bit(q).index for q in inst.qubits]
                name = op.name
                if name == "h":
                    _apply_single(state, qs[0], np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2))
                elif name == "x":
                    _apply_single(state, qs[0], np.array([[0, 1], [1, 0]], dtype=complex))
                elif name == "z":
                    _apply_single(state, qs[0], np.array([[1, 0], [0, -1]], dtype=complex))
                elif name == "s":
                    _apply_single(state, qs[0], np.array([[1, 0], [0, 1j]], dtype=complex))
                elif name == "sdg":
                    _apply_single(state, qs[0], np.array([[1, 0], [0, -1j]], dtype=complex))
                elif name in ("cx", "cnot"):
                    _apply_cx(state, qs[0], qs[1])
                elif name in ("mcx", "mcx_gray", "mcx_recursive", "mcx_v-chain"):
                    _apply_mcx(state, qs[:-1], qs[-1])
                elif name == "oracle":
                    if rng.random() >= p_misfire:
                        n, m = oracle.n, oracle.m
                        xmask, ymask = (1 << n) - 1, (1 << m) - 1
                        new = np.zeros_like(state)
                        for idx, amp in enumerate(state):
                            x = idx & xmask
                            y = (idx >> n) & ymask
                            out = y ^ int(oracle.table[x])
                            new[x | (out << n)] += amp
                        state = new
                elif name == "measure":
                    c = circuit.find_bit(inst.clbits[0]).index
                    measured[c] = qs[0]
                elif name == "barrier":
                    continue
                else:
                    raise NotImplementedError(f"unsupported gate in local simulator: {name}")
            probs = np.abs(state) ** 2
            probs /= probs.sum()
            basis = int(rng.choice(len(probs), p=probs))
            bits = []
            for c in range(nc):
                q = measured.get(c)
                if q is None:
                    bits.append("0")
                    continue
                bit = (basis >> q) & 1
                if rng.random() < p_readout:
                    bit ^= 1
                bits.append(str(bit))
            results["".join(reversed(bits))] += 1
        return dict(results)
    return run
