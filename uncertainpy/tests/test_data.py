import unittest
import shutil
import os

import numpy as np

from uncertainpy import Data

class TestData(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


        self.data = Data()


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)



    def test_sortFeatures(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.data.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)
        features_0d, features_1d, features_2d = self.uncertainty.data.sortFeatures(results[0])

        self.assertIn("directComparison", features_1d)
        self.assertIn("feature2d", features_2d)
        self.assertIn("feature1d", features_1d)
        self.assertIn("feature0d", features_0d)
        self.assertIn("featureInvalid", features_0d)
