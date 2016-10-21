import unittest
import chaospy as cp
from uncertainpy import Distribution

class TestDistribution(unittest.TestCase):
    def setUp(self):
        def distribution(parameter, interval):
            return cp.Uniform(parameter - interval, parameter + interval)

        self.distribution = Distribution(0.1, distribution)

    def test_intit(self):
        distribution = Distribution(0.1)

        self.assertEqual(distribution.interval, 0.1)

    def test_intitFunction(self):
        def distribution(parameter, interval):
            return cp.Uniform(parameter - interval, parameter + interval)

        distribution = Distribution(0.1, distribution)

        self.assertEqual(distribution.interval, 0.1)
        self.assertIsInstance(distribution.function(120, distribution.interval), cp.Dist)


    def test_call(self):
        result = self.distribution(120)

        self.assertIsInstance(result, cp.Dist)


    def test_normal(self):
        result = self.distribution.normal(120)

        self.assertIsInstance(result, cp.Dist)

    def test_uniform(self):
        result = self.distribution.uniform(120)

        self.assertIsInstance(result, cp.Dist)


if __name__ == "__main__":
    unittest.main()
