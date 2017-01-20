import numpy as np
import unittest
import scipy.interpolate
import os
import shutil


from uncertainpy import RunModel
from uncertainpy.parameters import Parameters
from uncertainpy.features import GeneralFeatures
from uncertainpy import Distribution

from features import TestingFeatures
from models import TestingModel0d, TestingModel1d, TestingModel2d
from models import TestingModel1dAdaptive


class TestUncertainty(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


        self.runmodel = RunModel(TestingModel1d(),
                                 features=TestingFeatures(),
                                 verbose_level="error",
                                 output_dir_data=self.output_test_dir,
                                 output_dir_figures=self.output_test_dir,
                                 seed=self.seed)



    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        RunModel(TestingModel1d())


    # def test_evaluateNodeFunctionList(self):
    #     nodes = [[0, 1], [1, 2], [2, 3]]
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     result = self.runmodel.evaluateNodeFunctionList(nodes)
    #
    #     self.assertEqual(len(result), 3)
    #     self.assertTrue(result[0][1])
    #     self.assertFalse(result[0][2])
    #     self.assertTrue(result[0][3], [0, 1])
    #     self.assertTrue(result[1][3], [1, 2])
    #     self.assertTrue(result[2][3], [2, 3])
    #     self.assertEqual(result[0][4], ["a", "b"])
    #     self.assertEqual(result[0][6], {"features_to_run": ["feature0d", "feature1d",
    #                                                         "feature2d", "featureInvalid"]})
    #
    #     nodes = [[0, 1], [0, 1], [0, 1]]
    #     result = self.runmodel.evaluateNodeFunctionList(nodes)
    #
    #     self.assertEqual(result[0], result[1])
    #     self.assertEqual(result[1], result[2])
    #
    #
    # def test_evaluateNodesSequentialModel0d(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel = UncertaintyEstimation(TestingModel0d(),
    #                                              features=TestingFeatures(),
    #                                              verbose_level="error")
    #
    #     self.CPUs = 1
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #
    #
    #     self.assertEqual(set(results[0].keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #
    # def test_evaluateNodesParallelModel0D(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel = UncertaintyEstimation(TestingModel0d(),
    #                                              features=TestingFeatures(),
    #                                              verbose_level="error")
    #     self.CPUs = 3
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #
    #     self.assertEqual(set(results[0].keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #
    # def test_evaluateNodesSequentialModel1d(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.CPUs = 1
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #
    #     self.assertEqual(set(results[0].keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #
    # def test_evaluateNodesParallelModel1D(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.CPUs = 3
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #
    #     self.assertEqual(set(results[0].keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    # def test_evaluateNodesSequentialModel2d(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel = UncertaintyEstimation(TestingModel2d(),
    #                                              features=TestingFeatures(),)
    #
    #     self.CPUs = 1
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #
    #     self.assertEqual(set(results[0].keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #
    # def test_evaluateNodesParallelModel2D(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel = UncertaintyEstimation(TestingModel2d(),
    #                                              features=TestingFeatures(),
    #                                              verbose_level="error")
    #     self.CPUs = 3
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #
    #     self.assertEqual(set(results[0].keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #
    # def test_evaluateNodesNotSupressOutput(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     self.runmodel.supress_model_output = False
    #     self.runmodel.evaluateNodes(nodes)
    #
    #
    # def test_evaluateNodesSupressOutput(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     self.runmodel.supress_model_output = True
    #     self.runmodel.evaluateNodes(nodes)
    #
    #
    # def test_evaluateNodesNotSupressGraphics(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     self.runmodel.supress_model_graphics = False
    #     self.runmodel.evaluateNodes(nodes)
    #
    #
    # def test_evaluateNodesSupressGraphics(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     self.runmodel.supress_model_supress_model_graphics = True
    #     self.runmodel.evaluateNodes(nodes)
    #
    #
    # def test_sortFeaturesFromResults(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     features_0d, features_1d, features_2d = self.runmodel.sortFeaturesFromResults(results[0])
    #
    #     self.assertIn("directComparison", features_1d)
    #     self.assertIn("feature2d", features_2d)
    #     self.assertIn("feature1d", features_1d)
    #     self.assertIn("feature0d", features_0d)
    #     self.assertIn("featureInvalid", features_0d)
    #
    #
    # def test_storeResultsModel1dFeaturesAll(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.assertEqual(set(self.runmodel.data.U.keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #     self.assertEqual(set(self.runmodel.data.t.keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #     self.assertIn("directComparison", self.runmodel.data.U.keys())
    #     self.assertTrue(np.array_equal(self.runmodel.data.t["directComparison"],
    #                                    np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["directComparison"][0],
    #                                    np.arange(0, 10) + 1))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["directComparison"][1],
    #                                    np.arange(0, 10) + 3))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["directComparison"][2],
    #                                    np.arange(0, 10) + 5))
    #
    #
    #     self.assertFeature0d(results)
    #     self.assertFeature1d(results)
    #     self.assertFeature2d(results)
    #     self.assertFeatureInvalid(results)
    #
    #
    # def test_storeResultsModel1dFeaturesAllAdaptive(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     self.runmodel.model.adaptive_model = True
    #     self.runmodel.data.feature_list = ["feature0d", "feature1d", "feature2d",
    #                                           "featureInvalid"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.assertEqual(set(self.runmodel.data.U.keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #     self.assertEqual(set(self.runmodel.data.t.keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "directComparison", "featureInvalid"]))
    #
    #     self.assertIn("directComparison", self.runmodel.data.U.keys())
    #     self.assertTrue(np.array_equal(self.runmodel.data.t["directComparison"],
    #                                    np.arange(0, 10)))
    #     self.assertTrue(np.allclose(self.runmodel.data.U["directComparison"][0],
    #                                 np.arange(0, 10) + 1))
    #     self.assertTrue(np.allclose(self.runmodel.data.U["directComparison"][1],
    #                                 np.arange(0, 10) + 3))
    #     self.assertTrue(np.allclose(self.runmodel.data.U["directComparison"][2],
    #                                 np.arange(0, 10) + 5))
    #
    #     self.assertFeature0d(results)
    #     self.assertFeature1d(results)
    #     self.assertFeature2d(results)
    #     self.assertFeatureInvalid(results)
    #
    #
    # def assertFeature0d(self, results):
    #     self.assertIn("feature0d", self.runmodel.data.U.keys())
    #     self.assertIsNone(self.runmodel.data.t["feature0d"])
    #     self.assertEqual(self.runmodel.data.U["feature0d"][0], 1)
    #     self.assertEqual(self.runmodel.data.U["feature0d"][1], 1)
    #     self.assertEqual(self.runmodel.data.U["feature0d"][2], 1)
    #
    #
    # def assertFeature1d(self, results):
    #     self.assertIn("feature1d", self.runmodel.data.U.keys())
    #     self.assertTrue(np.array_equal(self.runmodel.data.t["feature1d"],
    #                                    np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["feature1d"][0],
    #                                    np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["feature1d"][1],
    #                                    np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["feature1d"][2],
    #                                    np.arange(0, 10)))
    #
    #
    # def assertFeature2d(self, results):
    #     self.assertIn("feature2d", self.runmodel.data.U.keys())
    #     self.assertTrue(np.array_equal(self.runmodel.data.t["feature2d"],
    #                                    np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["feature2d"][0],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["feature2d"][1],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(self.runmodel.data.U["feature2d"][2],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #
    #
    # def assertFeatureInvalid(self, results):
    #     self.assertIn("featureInvalid", self.runmodel.data.U.keys())
    #     self.assertIsNone(self.runmodel.data.t["featureInvalid"])
    #     self.assertTrue(np.isnan(self.runmodel.data.U["featureInvalid"][0]))
    #     self.assertTrue(np.isnan(self.runmodel.data.U["featureInvalid"][1]))
    #     self.assertTrue(np.isnan(self.runmodel.data.U["featureInvalid"][2]))
    #
    #
    #
    #
    # def test_performInterpolation(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     self.runmodel.model.adaptive_model = True
    #     self.runmodel.data.feature_list = []
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     ts = []
    #     interpolation = []
    #
    #     for solved in results:
    #         ts.append(solved["directComparison"][0])
    #         interpolation.append(solved["directComparison"][2])
    #
    #     self.assertTrue(np.array_equal(ts[0], ts[1]))
    #     self.assertTrue(np.array_equal(ts[1], ts[2]))
    #
    #     self.assertIsInstance(interpolation[0],
    #                           scipy.interpolate.fitpack2.UnivariateSpline)
    #     self.assertIsInstance(interpolation[1],
    #                           scipy.interpolate.fitpack2.UnivariateSpline)
    #     self.assertIsInstance(interpolation[2],
    #                           scipy.interpolate.fitpack2.UnivariateSpline)
    #
    #     t, interpolated_solves = self.runmodel.performInterpolation(ts, interpolation)
    #
    #     self.assertTrue(np.array_equal(t, np.arange(0, 10)))
    #     self.assertTrue(np.allclose(interpolated_solves[0],
    #                                 np.arange(0, 10) + 1))
    #     self.assertTrue(np.allclose(interpolated_solves[1],
    #                                 np.arange(0, 10) + 3.))
    #     self.assertTrue(np.allclose(interpolated_solves[2],
    #                                 np.arange(0, 10) + 5.))
    #
    #     ts[1] = np.arange(0, 20)
    #
    #     t, interpolated_solves = self.runmodel.performInterpolation(ts, interpolation)
    #
    #     self.assertTrue(np.array_equal(t, np.arange(0, 20)))
    #     self.assertTrue(np.allclose(interpolated_solves[0],
    #                                 np.arange(0, 20) + 1))
    #     self.assertTrue(np.allclose(interpolated_solves[1],
    #                                 np.arange(0, 20) + 3.))
    #     self.assertTrue(np.allclose(interpolated_solves[2],
    #                                 np.arange(0, 20) + 5.))
    #
    #
    #
    # def test_createMaskDirectComparison(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "directComparison")
    #
    #     self.assertEqual(len(masked_U), 3)
    #     self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
    #     self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 3))
    #     self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10) + 5))
    #     self.assertTrue(np.array_equal(nodes, masked_nodes))
    #
    #
    # def test_createMaskDirectComparisonNaN(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.runmodel.data.U["directComparison"][1] = np.nan
    #
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "directComparison")
    #
    #
    #     self.assertEqual(len(masked_U), 2)
    #     self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
    #     self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 5))
    #     self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
    #
    #
    # def test_createMaskWarning(self):
    #     if os.path.isdir(self.output_test_dir):
    #         shutil.rmtree(self.output_test_dir)
    #     os.mkdir(self.output_test_dir)
    #
    #     logfile = os.path.join(self.output_test_dir, "test.log")
    #
    #     self.runmodel = UncertaintyEstimation(TestingModel1d(),
    #                                              features=TestingFeatures(),
    #                                              verbose_level="warning",
    #                                              verbose_filename=logfile)
    #
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #     self.runmodel.verbose_filename = logfile
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.runmodel.data.U["directComparison"][1] = np.nan
    #
    #     self.runmodel.createMask(nodes, "directComparison")
    #
    #     message = "WARNING - uncertainty - Feature: directComparison does not yield results for all parameter combinations"
    #     self.assertTrue(message in open(logfile).read())
    #
    #     shutil.rmtree(self.output_test_dir)
    #
    #
    # def test_createMaskFeature0d(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature0d")
    #     self.assertIn("feature0d", self.runmodel.data.U.keys())
    #     self.assertEqual(masked_U[0], 1)
    #     self.assertEqual(masked_U[1], 1)
    #     self.assertEqual(masked_U[2], 1)
    #     self.assertTrue(np.array_equal(nodes, masked_nodes))
    #
    #
    # def test_createMaskFeature0dNan(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.runmodel.data.U["feature0d"] = np.array([1, np.nan, 1])
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature0d")
    #
    #     self.assertTrue(np.array_equal(masked_U, np.array([1, 1])))
    #     self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
    #
    #
    #
    # def test_createMaskFeature1d(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     # feature1d
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature1d")
    #     self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10)))
    #     self.assertTrue(np.array_equal(nodes, masked_nodes))
    #
    #     # feature2d
    #
    # def test_createMaskFeature1dNan(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.runmodel.data.U["feature1d"] = np.array([np.arange(0, 10), np.nan, np.arange(0, 10)])
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature1d")
    #
    #     self.assertTrue(np.array_equal(masked_U, np.array([np.arange(0, 10), np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
    #
    #
    #
    # def test_totalSensitivity1(self):
    #     self.runmodel.data.sensitivity_1 = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2]}
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     self.runmodel.totalSensitivity(sensitivity="sensitivity_1")
    #
    #     self.assertEqual(self.runmodel.data.total_sensitivity_1["test2D"][0], 1/3.)
    #     self.assertEqual(self.runmodel.data.total_sensitivity_1["test2D"][1], 2/3.)
    #     self.assertEqual(self.runmodel.data.total_sensitivity_1["test1D"][0], 1/3.)
    #     self.assertEqual(self.runmodel.data.total_sensitivity_1["test1D"][1], 2/3.)
    #
    #
    # def test_totalSensitivityT(self):
    #     self.runmodel.data.sensitivity_t = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2]}
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     self.runmodel.totalSensitivity(sensitivity="sensitivity_t")
    #
    #     self.assertEqual(self.runmodel.data.total_sensitivity_t["test2D"][0], 1/3.)
    #     self.assertEqual(self.runmodel.data.total_sensitivity_t["test2D"][1], 2/3.)
    #     self.assertEqual(self.runmodel.data.total_sensitivity_t["test1D"][0], 1/3.)
    #     self.assertEqual(self.runmodel.data.total_sensitivity_t["test1D"][1], 2/3.)
    #
    #
    # def test_createMaskFeature2d(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature2d")
    #     self.assertTrue(np.array_equal(masked_U[0],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(masked_U[1],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(masked_U[2],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(nodes, masked_nodes))
    #
    #
    #
    # def test_createMaskFeature2dNan(self):
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.runmodel.data.U["feature2d"][1] = np.nan
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature2d")
    #
    #     self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
    #
    #
    #     self.assertEqual(len(masked_U), 2)
    #     self.assertTrue(np.array_equal(masked_U[0],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(masked_U[1],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #
    #
    # def test_createMaskFeature2dnodes1DNaN(self):
    #     nodes = np.array([0, 1, 2])
    #
    #     self.runmodel.data.uncertain_parameters = ["a"]
    #
    #     results = self.runmodel.evaluateNodes(nodes)
    #     self.runmodel.storeResults(results)
    #
    #     self.runmodel.data.U["feature2d"][1] = np.nan
    #     masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature2d")
    #
    #     self.assertTrue(np.array_equal(masked_nodes, np.array([0, 2])))
    #
    #     self.assertEqual(len(masked_U), 2)
    #     self.assertTrue(np.array_equal(masked_U[0],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
    #     self.assertTrue(np.array_equal(masked_U[1],
    #                                    np.array([np.arange(0, 10),
    #                                              np.arange(0, 10)])))
