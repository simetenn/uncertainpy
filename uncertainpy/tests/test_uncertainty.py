import numpy as np
import unittest
import scipy.interpolate
import warnings

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


    def test_storeResultsModel1dFeaturesAllAdaptive(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.adaptive_model = True
        self.uncertainty.feature_list = ["feature0d", "feature1d", "feature2d",
                                         "featureInvalid"]

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
        self.assertTrue(np.allclose(self.uncertainty.U["directComparison"][0],
                                    np.arange(0, 10) + 1))
        self.assertTrue(np.allclose(self.uncertainty.U["directComparison"][1],
                                    np.arange(0, 10) + 3))
        self.assertTrue(np.allclose(self.uncertainty.U["directComparison"][2],
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
        self.assertTrue(np.array_equal(self.uncertainty.U["feature1d"][1],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature1d"][2],
                                       np.arange(0, 10)))


    def assertFeature2d(self, results):
        self.assertIn("feature2d", self.uncertainty.U.keys())
        self.assertTrue(np.array_equal(self.uncertainty.t["feature2d"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature2d"][0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature2d"][1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(self.uncertainty.U["feature2d"][2],
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
        self.uncertainty.feature_list = []

        results = self.uncertainty.evaluateNodes(nodes)
        ts = []
        interpolation = []

        for solved in results:
            ts.append(solved["directComparison"][0])
            interpolation.append(solved["directComparison"][2])

        self.assertTrue(np.array_equal(ts[0], ts[1]))
        self.assertTrue(np.array_equal(ts[1], ts[2]))

        self.assertIsInstance(interpolation[0],
                              scipy.interpolate.fitpack2.UnivariateSpline)
        self.assertIsInstance(interpolation[1],
                              scipy.interpolate.fitpack2.UnivariateSpline)
        self.assertIsInstance(interpolation[2],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        t, interpolated_solves = self.uncertainty.performInterpolation(ts, interpolation)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 10) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 10) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 10) + 5.))

        ts[1] = np.arange(0, 20)

        t, interpolated_solves = self.uncertainty.performInterpolation(ts, interpolation)

        self.assertTrue(np.array_equal(t, np.arange(0, 20)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 20) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 20) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 20) + 5.))



    def test_createMaskDirectComparison(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.warning_flag = False

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        masked_nodes, masked_U = self.uncertainty.createMask(nodes, "directComparison")

        self.assertEqual(len(masked_U), 3)
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(nodes, masked_nodes))

        ### Todo working here
        # add specific test cases, set one part of the masked array to Nan


        # feature0d
    def test_createMaskFeature0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.warning_flag = False

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        masked_nodes, masked_U = self.uncertainty.createMask(nodes, "feature0d")
        self.assertIn("feature0d", self.uncertainty.U.keys())
        self.assertEqual(masked_U[0], 1)
        self.assertEqual(masked_U[1], 1)
        self.assertEqual(masked_U[2], 1)
        self.assertTrue(np.array_equal(nodes, masked_nodes))


    def test_createMaskFeature0dNan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.warning_flag = False

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        self.uncertainty.U["feature0d"] = np.array([1, np.nan, 1])
        self.uncertainty.U["feature0d"] = np.ma.masked_invalid(self.uncertainty.U["feature0d"])
        masked_nodes, masked_U = self.uncertainty.createMask(nodes, "feature0d")

        self.assertTrue(np.array_equal(masked_U, np.array([1, 1])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))





    def test_createMaskFeature1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.warning_flag = False

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        # feature1d
        masked_nodes, masked_U = self.uncertainty.createMask(nodes, "feature1d")
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10)))
        self.assertTrue(np.array_equal(nodes, masked_nodes))

        # feature2d

    def test_createMaskFeature1dNan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.warning_flag = False

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        # TODO Working here

        self.uncertainty.U["feature1d"] = [np.arange(0, 10), [np.nan], np.arange(0, 10)]
        print self.uncertainty.U["feature1d"]
        print np.isnan(self.uncertainty.U["feature1d"])
        #print np.ma.masked_array(self.uncertainty.U["feature1d"], np.isnan(self.uncertainty.U["feature1d"]))
        self.uncertainty.U["feature1d"] = np.ma.masked_where(np.isnan(self.uncertainty.U["feature1d"]), self.uncertainty.U["feature1d"])
        masked_nodes, masked_U = self.uncertainty.createMask(nodes, "feature1d")

        print masked_U

        self.assertTrue(np.array_equal(masked_U, np.array([np.arange(0, 10), np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



    def test_createMaskFeature2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.uncertainty.uncertain_parameters = ["a", "b"]
        self.uncertainty.warning_flag = False

        results = self.uncertainty.evaluateNodes(nodes)
        self.uncertainty.storeResults(results)

        masked_nodes, masked_U = self.uncertainty.createMask(nodes, "feature2d")
        self.assertTrue(np.array_equal(masked_U[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_U[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_U[2],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(nodes, masked_nodes))



    # def test_createMaskFeature2dNan(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.uncertainty.uncertain_parameters = ["a", "b"]
    #     self.uncertainty.warning_flag = False
    #
    #     results = self.uncertainty.evaluateNodes(nodes)
    #     self.uncertainty.storeResults(results)
    #
    #
    #     # print self.uncertainty.U["feature2d"][1]
    #     self.uncertainty.U["feature2d"][1] = np.nan
    #     # print "-----------"
    #     # print self.uncertainty.U["feature2d"]
    #     self.uncertainty.U["feature2d"] = np.ma.masked_invalid(self.uncertainty.U["feature2d"])
    #     # print "-----------"
    #     # print self.uncertainty.U["feature2d"]
    #
    #     masked_nodes, masked_U = self.uncertainty.createMask(nodes, "feature2d")
    #
    #     # print masked_U
    #     # print masked_nodes
    #
    #     self.assertTrue(np.array_equal(masked_U, np.array([[np.arange(0, 10), np.arange(0, 10)],
    #                                                        [np.arange(0, 10), np.arange(0, 10)]])))
    #     self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
    #
    #






if __name__ == "__main__":
    unittest.main()
