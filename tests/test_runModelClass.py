import numpy as np
import unittest
import scipy.interpolate
import os
import shutil


from uncertainpy import RunModel

from features import TestingFeatures
from models import TestingModel0d, TestingModel1d, TestingModel2d


class TestRunModelClass(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


        self.runmodel = RunModel(TestingModel1d(),
                                 features=TestingFeatures(),
                                 supress_model_output=True,
                                 supress_model_graphics=True)



    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        RunModel(TestingModel1d())


    def test_evaluateNodeFunctionList(self):
        nodes = [[0, 1], [1, 2], [2, 3]]
        self.runmodel.data.uncertain_parameters = ["a", "b"]
        result = self.runmodel.evaluateNodeFunctionList(nodes)

        self.assertEqual(len(result), 3)
        self.assertTrue(result[0][1])
        self.assertFalse(result[0][2])
        self.assertTrue(result[0][3], [0, 1])
        self.assertTrue(result[1][3], [1, 2])
        self.assertTrue(result[2][3], [2, 3])
        self.assertEqual(result[0][4], ["a", "b"])
        self.assertEqual(result[0][6], {"features_to_run": ["feature0d", "feature1d",
                                                            "feature2d", "featureInvalid"]})

        nodes = [[0, 1], [0, 1], [0, 1]]
        result = self.runmodel.evaluateNodeFunctionList(nodes)

        self.assertEqual(result[0], result[1])
        self.assertEqual(result[1], result[2])


    def test_evaluateNodesSequentialModel0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel = RunModel(TestingModel0d(), features=TestingFeatures(), CPUs=1)

        self.runmodel.data.uncertain_parameters = ["a", "b"]

        results = self.runmodel.evaluateNodes(nodes)


        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesParallelModel0D(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel = RunModel(TestingModel0d(), features=TestingFeatures(), CPUs=3)
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        results = self.runmodel.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesSequentialModel1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.CPUs = 1
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        results = self.runmodel.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesParallelModel1D(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.CPUs = 3
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        results = self.runmodel.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

    def test_evaluateNodesSequentialModel2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel = RunModel(TestingModel2d(), features=TestingFeatures(), CPUs=1)

        self.runmodel.data.uncertain_parameters = ["a", "b"]

        results = self.runmodel.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesParallelModel2D(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel = RunModel(TestingModel2d(), features=TestingFeatures(), CPUs=3)

        self.runmodel.data.uncertain_parameters = ["a", "b"]

        results = self.runmodel.evaluateNodes(nodes)

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))


    def test_evaluateNodesNotSupressOutput(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.data.uncertain_parameters = ["a", "b"]
        self.runmodel.supress_model_output = False
        self.runmodel.evaluateNodes(nodes)


    def test_evaluateNodesSupressOutput(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.data.uncertain_parameters = ["a", "b"]
        self.runmodel.supress_model_output = True
        self.runmodel.evaluateNodes(nodes)


    def test_evaluateNodesNotSupressGraphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.data.uncertain_parameters = ["a", "b"]
        self.runmodel.supress_model_graphics = False
        self.runmodel.evaluateNodes(nodes)


    def test_evaluateNodesSupressGraphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.data.uncertain_parameters = ["a", "b"]
        self.runmodel.supress_model_supress_model_graphics = True
        self.runmodel.evaluateNodes(nodes)



    def test_storeResultsModel1dFeaturesAll(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        results = self.runmodel.evaluateNodes(nodes)

        self.runmodel.storeResults(results)

        self.assertEqual(set(self.runmodel.data.U.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertEqual(set(self.runmodel.data.t.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertIn("directComparison", self.runmodel.data.U.keys())
        self.assertTrue(np.array_equal(self.runmodel.data.t["directComparison"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.runmodel.data.U["directComparison"][0],
                                       np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(self.runmodel.data.U["directComparison"][1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(self.runmodel.data.U["directComparison"][2],
                                       np.arange(0, 10) + 5))


        self.assertFeature0d(self.runmodel.data)
        self.assertFeature1d(self.runmodel.data)
        self.assertFeature2d(self.runmodel.data)
        self.assertFeatureInvalid(self.runmodel.data)


    def test_storeResultsModel1dFeaturesAllAdaptive(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.data.uncertain_parameters = ["a", "b"]
        self.runmodel.model.adaptive_model = True
        self.runmodel.data.feature_list = ["feature0d", "feature1d", "feature2d",
                                           "featureInvalid"]

        results = self.runmodel.evaluateNodes(nodes)
        self.runmodel.storeResults(results)

        self.assertEqual(set(self.runmodel.data.U.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertEqual(set(self.runmodel.data.t.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertIn("directComparison", self.runmodel.data.U.keys())
        self.assertTrue(np.array_equal(self.runmodel.data.t["directComparison"],
                                       np.arange(0, 10)))
        self.assertTrue(np.allclose(self.runmodel.data.U["directComparison"][0],
                                    np.arange(0, 10) + 1))
        self.assertTrue(np.allclose(self.runmodel.data.U["directComparison"][1],
                                    np.arange(0, 10) + 3))
        self.assertTrue(np.allclose(self.runmodel.data.U["directComparison"][2],
                                    np.arange(0, 10) + 5))

        self.assertFeature0d(self.runmodel.data)
        self.assertFeature1d(self.runmodel.data)
        self.assertFeature2d(self.runmodel.data)
        self.assertFeatureInvalid(self.runmodel.data)


    def assertFeature0d(self, data):
        self.assertIn("feature0d", self.runmodel.data.U.keys())
        self.assertIsNone(data.t["feature0d"])
        self.assertEqual(data.U["feature0d"][0], 1)
        self.assertEqual(data.U["feature0d"][1], 1)
        self.assertEqual(data.U["feature0d"][2], 1)


    def assertFeature1d(self, data):
        self.assertIn("feature1d", data.U.keys())
        self.assertTrue(np.array_equal(data.t["feature1d"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data.U["feature1d"][0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data.U["feature1d"][1],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data.U["feature1d"][2],
                                       np.arange(0, 10)))


    def assertFeature2d(self, data):
        self.assertIn("feature2d", data.U.keys())
        self.assertTrue(np.array_equal(data.t["feature2d"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data.U["feature2d"][0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(data.U["feature2d"][1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(data.U["feature2d"][2],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))


    def assertFeatureInvalid(self, data):
        self.assertIn("featureInvalid", data.U.keys())
        self.assertIsNone(data.t["featureInvalid"])
        self.assertTrue(np.isnan(data.U["featureInvalid"][0]))
        self.assertTrue(np.isnan(data.U["featureInvalid"][1]))
        self.assertTrue(np.isnan(data.U["featureInvalid"][2]))


    def test_isSolvesAdaptiveFalse(self):
        test_solves = [{"directComparison": (None, np.arange(0, 10), None),
                        "feature2d": (None, np.array([np.arange(0, 10),
                                                      np.arange(0, 10)]), None),
                        "feature1d": (None, np.arange(0, 10), None),
                        "feature0d": (None, 1, None),
                        "featureInvalid": (None, np.nan, None)},
                       {"directComparison": (None, np.arange(0, 10), None),
                        "feature2d": (None, np.array([np.arange(0, 10),
                                                      np.arange(0, 10)]), None),
                        "feature1d": (None, np.arange(0, 10), None),
                        "feature0d": (None, 1, None),
                        "featureInvalid": (None, np.nan, None)},
                       {"directComparison": (None, np.arange(0, 10), None),
                        "feature2d": (None, np.array([np.arange(0, 10),
                                                      np.arange(0, 10)]), None),
                        "feature1d": (None, np.arange(0, 10), None),
                        "feature0d": (None, 1, None),
                        "featureInvalid": (None, np.nan, None)}]

        self.runmodel.data.features_1d = ["directComparison", "feature1d"]
        self.runmodel.data.features_2d = ["feature2d"]


        self.assertFalse(self.runmodel.isSolvesAdaptive(test_solves))



    def test_isSolvesAdaptiveTrue(self):


        test_solves = [{"directComparison": (None, np.arange(0, 10), None),
                        "feature2d": (None, np.array([np.arange(0, 10),
                                                      np.arange(0, 10)]), None),
                        "feature1d": (None, np.arange(0, 10), None),
                        "feature0d": (None, 1, None),
                        "featureInvalid": (None, np.nan, None)},
                       {"directComparison": (None, np.arange(0, 10), None),
                        "feature2d": (None, np.array([np.arange(0, 10),
                                                      np.arange(0, 10)]), None),
                        "feature1d": (None, np.arange(0, 10), None),
                        "feature0d": (None, 1, None),
                        "featureInvalid": (None, np.nan, None)},
                       {"directComparison": (None, np.arange(0, 10), None),
                        "feature2d": (None, np.array([np.arange(0, 10),
                                                      np.arange(0, 10)]), None),
                        "feature1d": (None, np.arange(0, 15), None),
                        "feature0d": (None, 1, None),
                        "featureInvalid": (None, np.nan, None)}]

        self.runmodel.data.features_1d = ["directComparison", "feature1d"]
        self.runmodel.data.features_2d = ["feature2d"]


        self.assertTrue(self.runmodel.isSolvesAdaptive(test_solves))


    def test_performInterpolation(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.data.uncertain_parameters = ["a", "b"]
        self.runmodel.model.adaptive_model = True
        self.runmodel.data.feature_list = []

        results = self.runmodel.evaluateNodes(nodes)
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

        t, interpolated_solves = self.runmodel.performInterpolation(ts, interpolation)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 10) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 10) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 10) + 5.))

        ts[1] = np.arange(0, 20)

        t, interpolated_solves = self.runmodel.performInterpolation(ts, interpolation)

        self.assertTrue(np.array_equal(t, np.arange(0, 20)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 20) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 20) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 20) + 5.))



    def test_run(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel = RunModel(TestingModel1d(), features=TestingFeatures(), CPUs=1)
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        data = self.runmodel.run(nodes)


        self.assertEqual(set(data.U.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertEqual(set(data.t.keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "directComparison", "featureInvalid"]))

        self.assertIn("directComparison", data.U.keys())
        self.assertTrue(np.array_equal(data.t["directComparison"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data.U["directComparison"][0],
                                       np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(data.U["directComparison"][1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(data.U["directComparison"][2],
                                       np.arange(0, 10) + 5))


        self.assertFeature0d(data)
        self.assertFeature1d(data)
        self.assertFeature2d(data)
        self.assertFeatureInvalid(data)
