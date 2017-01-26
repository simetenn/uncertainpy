import numpy as np
import unittest
import scipy.interpolate
import chaospy as cp
import os
import shutil
import subprocess
import scipy.interpolate


from uncertainpy import UncertaintyCalculations
from uncertainpy.parameters import Parameters
from uncertainpy.features import GeneralFeatures
from uncertainpy import Distribution
from uncertainpy import RunModel

from features import TestingFeatures
from models import TestingModel0d, TestingModel1d, TestingModel2d
from models import TestingModel1dAdaptive


class TestUncertaintyCalculations(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=features,
                                                                verbose_level="error",
                                                                seed=self.seed)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        uncertainty = UncertaintyCalculations(TestingModel1d())

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.features, GeneralFeatures)





    def test_intitFeatures(self):
        uncertainty_calculations = UncertaintyCalculations(TestingModel1d(),
                                                           verbose_level="error")
        self.assertIsInstance(uncertainty_calculations.features, GeneralFeatures)

        uncertainty_calculations = UncertaintyCalculations(TestingModel1d(),
                                                           features=TestingFeatures(),
                                                           verbose_level="error")
        self.assertIsInstance(uncertainty_calculations.features, TestingFeatures)


    def test_createDistributionNone(self):

        self.uncertainty_calculations.createDistribution()

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)

    def test_createDistributionString(self):

        self.uncertainty_calculations.createDistribution("a")

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)

    def test_createDistributionList(self):

        self.uncertainty_calculations.createDistribution(["a"])

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)


    def test_createMaskDirectComparison(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "directComparison")

        self.assertEqual(len(masked_U), 3)
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(nodes, masked_nodes))


    def test_createMaskDirectComparisonNaN(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["directComparison"][1] = np.nan

        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "directComparison")


        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


    def test_createMaskWarning(self):
        logfile = os.path.join(self.output_test_dir, "test.log")

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])


        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=features,
                                                                verbose_level="warning",
                                                                verbose_filename=logfile,
                                                                seed=self.seed)

        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["directComparison"][1] = np.nan

        self.uncertainty_calculations.createMask(nodes, "directComparison")

        message = "WARNING - uncertainty_calculations - Feature: directComparison does not yield results for all parameter combinations"
        self.assertTrue(message in open(logfile).read())


    def test_createMaskFeature0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature0d")

        self.assertEqual(masked_U[0], 1)
        self.assertEqual(masked_U[1], 1)
        self.assertEqual(masked_U[2], 1)
        self.assertTrue(np.array_equal(nodes, masked_nodes))


    def test_createMaskFeature0dNan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["feature0d"] = np.array([1, np.nan, 1])
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature0d")

        self.assertTrue(np.array_equal(masked_U, np.array([1, 1])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



    def test_createMaskFeature1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        # feature1d
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature1d")
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10)))
        self.assertTrue(np.array_equal(nodes, masked_nodes))

        # feature2d

    def test_createMaskFeature1dNan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        self.uncertainty_calculations.data.U["feature1d"] = np.array([np.arange(0, 10), np.nan, np.arange(0, 10)])
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature1d")

        self.assertTrue(np.array_equal(masked_U, np.array([np.arange(0, 10), np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



    def test_createMaskFeature2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature2d")
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



    def test_createMaskFeature2dNan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["feature2d"][1] = np.nan
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature2d")

        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_U[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))


    def test_createMaskFeature2dnodes1DNaN(self):
        nodes = np.array([0, 1, 2])
        uncertain_parameters = ["a"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        self.uncertainty_calculations.data.U["feature2d"][1] = np.nan
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature2d")

        self.assertTrue(np.array_equal(masked_nodes, np.array([0, 2])))

        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_U[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))



    def test_convertUncertainParametersList(self):
        result = self.uncertainty_calculations.convertUncertainParameters(["a", "b"])

        self.assertEqual(result, ["a", "b"])

    def test_convertUncertainParametersString(self):
        result = self.uncertainty_calculations.convertUncertainParameters("a")

        self.assertEqual(result, ["a"])

    def test_convertUncertainParametersNone(self):
            result = self.uncertainty_calculations.convertUncertainParameters(None)

            self.assertEqual(result, ["a", "b"])


    def test_PCERegressionAll(self):

        self.uncertainty_calculations.PCERegression()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["directComparison"], cp.Poly)


    def test_PCERegressionOne(self):

        self.uncertainty_calculations.PCERegression("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["directComparison"], cp.Poly)

    #

    def test_PCERegressionAdaptiveError(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1dAdaptive(parameters, adaptive_model=False)
        model.setAllDistributions(Distribution(1).uniform)

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=features,
                                                                verbose_level="debug",
                                                                supress_model_output=False)

        with self.assertRaises(ValueError):
            self.uncertainty_calculations.PCERegression()



    def test_PCERegressionRosenblattAll(self):

        self.uncertainty_calculations.PCERegressionRosenblatt()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["directComparison"], cp.Poly)


    def test_PCERegressionRosenblattOne(self):

        self.uncertainty_calculations.PCERegressionRosenblatt("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["directComparison"], cp.Poly)

    # def test_PCERquadrature(self):
    #
    #     self.uncertainty_calculations.PCEQuadrature()
    #
    #     print self.uncertainty_calculations.U_hat
    #     self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["directComparison"], cp.Poly)

    def test_totalSensitivity1(self):
        self.runmodel.data.sensitivity_1 = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2]}
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        self.runmodel.totalSensitivity(sensitivity="sensitivity_1")

        self.assertEqual(self.runmodel.data.total_sensitivity_1["test2D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_1["test2D"][1], 2/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_1["test1D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_1["test1D"][1], 2/3.)


    def test_totalSensitivityT(self):
        self.runmodel.data.sensitivity_t = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2]}
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        self.runmodel.totalSensitivity(sensitivity="sensitivity_t")

        self.assertEqual(self.runmodel.data.total_sensitivity_t["test2D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_t["test2D"][1], 2/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_t["test1D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_t["test1D"][1], 2/3.)
