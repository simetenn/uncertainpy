import unittest
import chaospy as cp
from uncertainpy import uniform, normal

class TestDistribution(unittest.TestCase):
    def test_normal(self):
        dist = normal(0.1)
        self.assertIsInstance(dist(5), cp.Dist)

    def test_uniform(self):
        dist = uniform(0.1)
        self.assertIsInstance(dist(5), cp.Dist)


    def test_normal_error(self):
        with self.assertRaises(ValueError):
            normal(0)

    def test_uniform_error(self):
        with self.assertRaises(ValueError):
            uniform(0)


if __name__ == "__main__":
    unittest.main()
