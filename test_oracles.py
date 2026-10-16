"""Fast tests for oracle definitions and the local simulator's pure data model."""
import unittest
from simulator import LocalOracle

class OracleDefinitionTests(unittest.TestCase):
    def test_linear_reference(self):
        o = LocalOracle("linear", a=178, b=1, seed=1)
        self.assertEqual([int(o.table[x]) for x in (0, 1, 2, 255)], [1, 1, 0, 1])

    def test_simon_reference(self):
        o = LocalOracle("simon", s=105, seed=1)
        self.assertEqual([int(o.table[x]) for x in (5, 108, 105, 200)], [5, 5, 0, 161])

    def test_simon2_reference(self):
        o = LocalOracle("simon2", H={0, 15, 49, 62}, seed=1)
        self.assertEqual([int(o.table[x]) for x in (0, 15, 49, 62, 1, 200)], [0, 0, 0, 0, 1, 199])

    def test_permutation_reference(self):
        o = LocalOracle("perm", affine_example=True, seed=1)
        self.assertEqual([int(o.table[x]) for x in (0, 1, 2, 255)], [11, 48, 85, 230])
        self.assertEqual(len(set(map(int, o.table))), 256)

    def test_balanced_has_128_ones(self):
        o = LocalOracle("balanced", seed=2)
        self.assertEqual(int(o.table.sum()), 128)

    def test_grover_has_one_marked_input(self):
        o = LocalOracle("grover", t=173, seed=2)
        self.assertEqual(int(o.table.sum()), 1)
        self.assertEqual(int(o.table[173]), 1)

    def test_simon2_secret_format(self):
        o = LocalOracle("simon2", H={0, 15, 49, 62}, seed=1)
        self.assertEqual(o.secret, [15, 49, 62])

if __name__ == "__main__":
    unittest.main()
