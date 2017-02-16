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
from uncertainpy import Data

from testing_modules import TestingFeatures
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
        parameters.setAllDistributions(Distribution(0.5).uniform)
        model = TestingModel1d(parameters)

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


    def test_set_model(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)
        model = TestingModel1d(parameters)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(features=features,
                                                                verbose_level="error",
                                                                seed=self.seed)

        self.uncertainty_calculations.set_model(model)

        self.assertIsInstance(self.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.model, TestingModel1d)


    def test_set_features(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)
        model = TestingModel1d(parameters)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(verbose_level="error",
                                                                seed=self.seed)

        self.uncertainty_calculations.set_features(features)

        self.assertIsInstance(self.uncertainty_calculations.features, TestingFeatures)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.features, TestingFeatures)

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


    def test_PCECustom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.PCECustom()


    def test_CustomUQ(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.CustomUQ()

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
        self.uncertainty_calculations.data = Data()
        self.uncertainty_calculations.data.sensitivity_1 = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2], "testNone": None}
        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.totalSensitivity(sensitivity="sensitivity_1")

        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test2D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test2D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test1D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test1D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["testNone"], None)


    def test_totalSensitivityT(self):
        self.uncertainty_calculations.data = Data()
        self.uncertainty_calculations.data.sensitivity_t = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2], "testNone": None}
        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.totalSensitivity(sensitivity="sensitivity_t")

        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test2D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test2D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test1D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test1D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["testNone"], None)


    def test_PCAnalysis(self):
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
                                                                verbose_level="error")

        self.uncertainty_calculations.data = Data()
        self.uncertainty_calculations.data.feature_list = ["feature0d", "feature1d",
                                                           "feature2d", "directComparison"]

        q0, q1 = cp.variable(2)
        parameter_space = model.parameters.getUncertain("distribution")
        self.uncertainty_calculations.distribution = cp.J(*parameter_space)

        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]


        self.uncertainty_calculations.U_hat["directComparison"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature0d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature1d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature2d"] = cp.Poly([q0, q1*q0, q1])

        self.uncertainty_calculations.PCAnalysis()


        # Test if all calculated properties actually exists
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.p_95["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_95["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_95["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_95["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.E["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.E["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.E["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.E["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.Var["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.Var["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.Var["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.Var["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["directComparison"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["feature0d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["feature1d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["feature2d"], list)


        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["directComparison"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["feature0d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["feature1d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["feature2d"], list)



        self.assertIsInstance(self.uncertainty_calculations.U_mc["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature2d"], np.ndarray)


    def test_PC(self):
        data = self.uncertainty_calculations.PC()

        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)



    def test_PCRosenblatt(self):
        data = self.uncertainty_calculations.PC(rosenblatt=True)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_Rosenblatt.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_Rosenblatt.h5")
        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)



    def test_PCCustom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.PC(method="custom")



    def test_PCError(self):
        with self.assertRaises(ValueError):
            self.uncertainty_calculations.PC(method="not implemented")


    def test_PCParameterA(self):
        data = self.uncertainty_calculations.PC("a")

        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-a.h5")
        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])

        self.assertEqual(result, 0)


    def test_PCParameterB(self):
        data = self.uncertainty_calculations.PC("b")

        filename = os.path.join(self.output_test_dir, "UncertaintyCalculations_single-parameter-b.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/UncertaintyCalculations_single-parameter-b.h5")
        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])

        self.assertEqual(result, 0)



    def test_MC(self):
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
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")



        data = self.uncertainty_calculations.MC()

        # Rough tests
        self.assertTrue(np.allclose(self.uncertainty_calculations.data.E["directComparison"],
                                    np.arange(0, 10) + 3, atol=0.1))
        self.assertTrue(np.allclose(self.uncertainty_calculations.data.Var["directComparison"],
                                    np.zeros(10), atol=0.1))
        self.assertTrue(np.all(np.less(self.uncertainty_calculations.data.p_05["directComparison"],
                                       np.arange(0, 10) + 3)))
        self.assertTrue(np.all(np.greater(self.uncertainty_calculations.data.p_95["directComparison"],
                                          np.arange(0, 10) + 3)))



        # Compare to pregenerated data
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)




    def test_MC_feature0d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)
        features = TestingFeatures(features_to_run=["feature0d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=features,
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")



        self.uncertainty_calculations.MC()

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.E["feature0d"],
                                       features.feature0d()))
        self.assertEqual(self.uncertainty_calculations.data.Var["feature0d"], 0)
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_05["feature0d"],
                                       features.feature0d()))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_95["feature0d"],
                                       features.feature0d()))


    def test_MC_feature1d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)
        features = TestingFeatures(features_to_run=["feature1d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=features,
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")



        self.uncertainty_calculations.MC()

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.E["feature1d"],
                                       features.feature1d()))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.Var["feature1d"],
                                       np.zeros(10)))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_05["feature1d"],
                                       features.feature1d()))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_95["feature1d"],
                                       features.feature1d()))



    def test_MC_feature2d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)
        features = TestingFeatures(features_to_run=["feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=features,
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")


        self.uncertainty_calculations.MC()

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.E["feature2d"],
                                       features.feature2d()))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.Var["feature2d"],
                                       np.zeros((2, 10))))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_05["feature2d"],
                                       features.feature2d()))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_95["feature2d"],
                                       features.feature2d()))
