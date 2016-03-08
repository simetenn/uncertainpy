import unittest

from uncertainpy import Distribution

class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.distribution = Distribution(0.1)
