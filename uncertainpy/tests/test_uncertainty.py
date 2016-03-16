import numpy as np
import os
import unittest

from uncertainpy import UncertaintyEstimation
from uncertainpy.features import TestingFeatures, NeuronFeatures
from uncertainpy.models import TestingModel0d, TestingModel1d, TestingModel2d


class TestUncertainty(unittest.TestCase):
    def setUp(self):
        self.uncertainty = UncertaintyEstimation(TestingModel1d(),
                                                 features=TestingFeatures(),
                                                 feature_list="all")


    def test_intit(self):
        uncertainty = UncertaintyEstimation(TestingModel1d())


    def test_intitFeatures(self):
        uncertainty = UncertaintyEstimation(TestingModel1d())
        self.assertIsInstance(uncertainty.features, NeuronFeatures)

        uncertainty = UncertaintyEstimation(TestingModel1d(),
                                            features=TestingFeatures())
        self.assertIsInstance(uncertainty.features, TestingFeatures)


    def test_intitFeatureList(self):
        uncertainty = UncertaintyEstimation(TestingModel1d(),
                                            features=TestingFeatures())
        self.assertEqual(uncertainty.feature_list, [])

        uncertainty = UncertaintyEstimation(TestingModel1d(),
                                            features=TestingFeatures(),
                                            feature_list="all")
        self.assertEqual(uncertainty.feature_list,
                         ["feature0d", "feature1d", "feature2d", "featureInvalid"])

        uncertainty = UncertaintyEstimation(TestingModel1d(),
                                            feature_list=["feature1", "feature2"])
        self.assertEqual(uncertainty.feature_list,
                         ["feature1", "feature2"])


    def test_resetValues(self):
        self.uncertainty.parameter_names = -1
        self.uncertainty.parameter_space = -1

        self.uncertainty.U = -1
        self.uncertainty.U_hat = -1
        self.uncertainty.distribution = -1
        self.uncertainty.solves = -1
        self.uncertainty.t = -1
        self.uncertainty.E = -1
        self.uncertainty.Var = -1
        self.uncertainty.U_mc = -1
        self.uncertainty.p_05 = -1
        self.uncertainty.p_95 = -1
        self.uncertainty.sensitivity = -1
        self.uncertainty.P = -1

        self.uncertainty.resetValues()

        self.assertIsNone(self.uncertainty.parameter_names)
        self.assertIsNone(self.uncertainty.parameter_space)
        self.assertEqual(self.uncertainty.U, {})
        self.assertEqual(self.uncertainty.U_hat, {})
        self.assertIsNone(self.uncertainty.distribution)
        self.assertIsNone(self.uncertainty.solves)
        self.assertEqual(self.uncertainty.t, {})
        self.assertEqual(self.uncertainty.E, {})
        self.assertEqual(self.uncertainty.Var, {})
        self.assertEqual(self.uncertainty.U_mc, {})
        self.assertEqual(self.uncertainty.p_05, {})
        self.assertEqual(self.uncertainty.p_95, {})
        self.assertEqual(self.uncertainty.sensitivity, {})
        self.assertIsNone(self.uncertainty.P)




    def test_evaluateNodeFunctionList(self):
        nodes = [[0, 1], [1, 2], [2, 3]]
        self.uncertainty.uncertain_parameters = ["a", "b"]
        result = self.uncertainty.evaluateNodeFunctionList(nodes)

        self.assertEqual(len(result), 3)
        self.assertTrue(result[0][1])
        self.assertFalse(result[0][2])
        self.assertTrue(result[0][3], [0, 1])
        self.assertTrue(result[1][3], [1, 2])
        self.assertTrue(result[2][3], [2, 3])
        self.assertEqual(result[0][4], ["a", "b"])
        self.assertEqual(result[0][5], ["feature0d", "feature1d",
                                        "feature2d", "featureInvalid"])
        self.assertEqual(result[0][7], {})

        nodes = [[0, 1], [0, 1], [0, 1]]
        result = self.uncertainty.evaluateNodeFunctionList(nodes)

        self.assertEqual(result[0], result[1])
        self.assertEqual(result[1], result[2])

        self.uncertainty.kwargs["feature_options"] = {"correct": True}
        self.uncertainty.kwargs["something_else"] = {"wrong": False}

        result = self.uncertainty.evaluateNodeFunctionList(nodes)
        self.assertEqual(result[0][7], {"correct": True})


    def test_evaluateNodesSequentialModel0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty = UncertaintyEstimation(TestingModel0d(),
                                                 features=TestingFeatures(),
                                                 feature_list="all")

        self.CPUs = 1
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesParallelModel0D(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty = UncertaintyEstimation(TestingModel0d(),
                                                 features=TestingFeatures(),
                                                 feature_list="all")
        self.CPUs = 3
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesSequentialModel1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.CPUs = 1
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesParallelModel1D(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.CPUs = 3
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

    def test_evaluateNodesSequentialModel2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty = UncertaintyEstimation(TestingModel2d(),
                                                 features=TestingFeatures(),
                                                 feature_list="all")

        self.CPUs = 1
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesParallelModel2D(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty = UncertaintyEstimation(TestingModel2d(),
                                                 features=TestingFeatures(),
                                                 feature_list="all")
        self.CPUs = 3
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesNotSupressOutput(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.supress_model_output = False
        self.uncertainty.evaluateNodes(nodes)


    def test_evaluateNodesSupressOutput(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.supress_model_output = True
        self.uncertainty.evaluateNodes(nodes)


    def test_evaluateNodesNotSupressGraphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.supress_model_graphics = False
        self.uncertainty.evaluateNodes(nodes)


    def test_evaluateNodesSupressGraphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.supress_model_supress_model_graphics = True
        self.uncertainty.evaluateNodes(nodes)



    def test_storeResultsModel1dFeaturesAll(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        self.assertEqual(set(self.uncertainty.U.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertEqual(set(self.uncertainty.t.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertIn("directComparison", self.uncertainty.U.keys())
        self.assertTrue(np.array_equal(self.uncertainty.t["directComparison"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["directComparison"][0],
                                       np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(self.uncertainty.U["directComparison"][1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(self.uncertainty.U["directComparison"][2],
                                       np.arange(0, 10) + 5))


        self.assertFeature0d(results)
        self.assertFeature1d(results)
        self.assertFeature2d(results)
        self.assertFeatureInvalid(results)


    def assertFeature0d(self, results):
        self.assertIn("feature0d", self.uncertainty.U.keys())
        self.assertIsNone(self.uncertainty.t["feature0d"])
        self.assertEqual(self.uncertainty.U["feature0d"][0], 1)
        self.assertEqual(self.uncertainty.U["feature0d"][1], 1)
        self.assertEqual(self.uncertainty.U["feature0d"][2], 1)


    def assertFeature1d(self, results):
        self.assertIn("feature1d", self.uncertainty.U.keys())
        self.assertTrue(np.array_equal(self.uncertainty.t["feature1d"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature1d"][0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature1d"][0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature1d"][0],
                                       np.arange(0, 10)))


    def assertFeature2d(self, results):
        self.assertIn("feature2d", self.uncertainty.U.keys())
        self.assertTrue(np.array_equal(self.uncertainty.t["feature2d"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature2d"][0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature2d"][0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature2d"][0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))


    def assertFeatureInvalid(self, results):
        self.assertIn("featureInvalid", self.uncertainty.U.keys())
        self.assertIsNone(self.uncertainty.t["featureInvalid"])
        self.assertIsInstance(self.uncertainty.U["featureInvalid"][0], np.ma.masked_array)
        self.assertEqual(self.uncertainty.U["featureInvalid"].count(), 0)


    def test_sortFeatures(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]

        results = self.uncertainty.evaluateNodes(nodes)
        features_0d, features_1d, features_2d = self.uncertainty.sortFeatures(results[0])

        self.assertIn("directComparison", features_1d)
        self.assertIn("feature2d", features_2d)
        self.assertIn("feature1d", features_1d)
        self.assertIn("feature0d", features_0d)
        self.assertIn("featureInvalid", features_0d)


    def test_performInterpolation(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.adaptive_model = True
        self.uncertainty.feature_list = ["feature0d", "feature1d", "featureInvalid"]

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)



    def atest_storeResultsModel1dFeaturesAllAdaptive(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.adaptive_model = True
        self.uncertainty.feature_list = ["feature0d", "feature1d", "featureInvalid"]

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        self.assertEqual(set(self.uncertainty.U.keys()),
                         set(["feature0d", "feature1d",
                              "directComparison", "featureInvalid"]))

        self.assertEqual(set(self.uncertainty.t.keys()),
                         set(["feature0d", "feature1d",
                              "directComparison", "featureInvalid"]))

        self.assertIn("directComparison", self.uncertainty.U.keys())
        self.assertTrue(np.array_equal(self.uncertainty.t["directComparison"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["directComparison"][0],
                                       np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(self.uncertainty.U["directComparison"][1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(self.uncertainty.U["directComparison"][2],
                                       np.arange(0, 10) + 5))




if __name__ == "__main__":
    unittest.main()
