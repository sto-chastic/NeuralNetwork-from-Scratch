import unittest
import numpy as np
from mlp_project.mlp import MLP

class TestMLP(unittest.TestCase):
    def test_forward_pass(self):
        mlp = MLP(2, 5, 3)
        result = mlp.propagate_forward([0.5, -0.3])
        self.assertEqual(len(result), 3)

if __name__ == "__main__":
    unittest.main()
