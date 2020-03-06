import unittest
import matrix
import random

import numpy as np


class TestMatrix(unittest.TestCase):

    def test_initialize(self):
        mat_a = matrix.random(2, 3)
        mat_b = matrix.ones(2, 3)
        mat_c = matrix.zeros(2, 3)

        assert mat_a.colunm == mat_b.colunm == mat_c.colunm == 3
        assert mat_a.row == mat_b.row == mat_c.row == 2

    def test_matmul(self):
        m = random.randint(2, 100)
        k = random.randint(2, 100)
        n = random.randint(2, 100)

        mat_a = matrix.random(m, k)
        mat_b = matrix.random(k, n)
        mat_c = mat_a * mat_b

        self.assertEqual(
            [mat_c.colunm, mat_c.row],
            [n, m]
        )

    def test_add(self):
        m = random.randint(2, 100)
        n = random.randint(2, 100)

        mat_a = matrix.random(m, n)
        mat_b = matrix.random(m, n)
        mat_c = mat_a + mat_b

        self.assertEqual(
            [mat_c.colunm, mat_c.row],
            [n, m]
        )

    def test_minus(self):
        m = random.randint(2, 100)
        n = random.randint(2, 100)

        mat_a = matrix.random(m, n)
        mat_b = matrix.random(m, n)
        mat_c = mat_a - mat_b

        self.assertEqual(
            [mat_c.colunm, mat_c.row],
            [n, m]
        )

    def test_attr(self):
        m = np.ones((10, 10))
        n = matrix.ones(10, 10)

        assert n.colunm == 10
        assert n.row == 10
        self.assertEqual(n.data, m.tolist())


if __name__ == "__main__":
    unittest.main()
