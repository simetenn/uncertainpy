import numpy as np
import os
import unittest

from uncertainpy.features import GeneralFeatures, GeneralNeuronFeatures
from uncertainpy.features import TestingFeatures, NeuronFeatures


class TestGeneralFeatures(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 1

        self.features = GeneralFeatures(t, U)


    def test_initNone(self):
        features = GeneralFeatures()


    def test_initUt(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 1

        features = GeneralFeatures(t, U)

        self.assertTrue(np.array_equal(features.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(features.U, np.arange(0, 10) + 1))



    def test_initUtility(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 1

        new_utility_methods = ["new"]

        features = GeneralFeatures(t, U, new_utility_methods)

        self.assertTrue(np.array_equal(features.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(features.U, np.arange(0, 10) + 1))
        self.assertIn("new", features.utility_methods)


    def test_cmd(self):
        result = self.features.cmd()

        self.assertEqual('general_features', result[1].split(".")[0])
        self.assertEqual('GeneralFeatures', result[2])


    def test_calculateFeatureNotImplemented(self):
        with self.assertRaises(AttributeError):
            self.features.calculateFeature("not_in_class")


    def test_calculateFeatureUtilityMethod(self):
        with self.assertRaises(TypeError):
            self.features.calculateFeature("cmd")


    def test_implementedFeatures(self):
        self.assertEqual(self.features.implementedFeatures(), [])


    def test_calculateAllFeatures(self):
        self.assertEqual(self.features.calculateAllFeatures(), {})



# class TestGeneralNeuronFeatures(unittest.TestCase):
#     def setUp(self):














if __name__ == "__main__":
    unittest.main()
