import numpy as np
import unittest
import chaospy as cp
import os
import shutil
import subprocess


from uncertainpy import UncertaintyCalculations
from uncertainpy.parameters import Parameters
from uncertainpy.features import GeneralFeatures
from uncertainpy import Distribution
from uncertainpy import Data
from uncertainpy.models import Model
from uncertainpy import SpikingFeatures


from testing_classes import TestingFeatures
from testing_classes import TestingModel1d, model_function
from testing_classes import TestingModelAdaptive


class TestUncertaintyCalculations(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10
        self.difference_treshold = 1e-10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.parameterlist = [["a", 1, None],
                              ["b", 2, None]]

        self.parameters = Parameters(self.parameterlist)
        self.parameters.set_all_distributions(Distribution(0.5).uniform)

        self.model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                                parameters=self.parameters,
                                                                features=features,
                                                                verbose_level="error",
                                                                seed=self.seed)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        uncertainty = UncertaintyCalculations(model=self.model,
                                              parameters=self.parameters)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.features, GeneralFeatures)



    def test_intit_features(self):
        uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                           verbose_level="error")
        self.assertIsInstance(uncertainty_calculations.features, GeneralFeatures)

        uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                           parameters=None,
                                                           features=TestingFeatures(),
                                                           verbose_level="error")
        self.assertIsInstance(uncertainty_calculations.features, TestingFeatures)


    def test_set_model(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error",
                                                                seed=self.seed)


        self.uncertainty_calculations.model = self.model

        self.assertIsInstance(self.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.model, TestingModel1d)


    def test_set_features(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error",
                                                                seed=self.seed)
        self.uncertainty_calculations.features = TestingFeatures()

        self.assertIsInstance(self.uncertainty_calculations.features, TestingFeatures)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.features, TestingFeatures)


    def test_feature_function(self):
        def feature_function(t, U):
                return "t", "U"

        self.uncertainty_calculations.features = feature_function
        self.assertIsInstance(self.uncertainty_calculations.features, GeneralFeatures)

        t, U = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        self.assertEqual(self.uncertainty_calculations.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(t, U):
                return "t", "U"

        def feature_function2(t, U):
                return "t2", "U2"


        self.uncertainty_calculations.features = [feature_function, feature_function2]
        self.assertIsInstance(self.uncertainty_calculations.features, GeneralFeatures)

        t, U = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")


        t, U = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.uncertainty_calculations.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

        self.assertEqual(self.uncertainty_calculations.features.features_to_run,
                         ["feature_function", "feature_function2"])


    def test_feature_functions_base(self):
        def feature_function(t, U):
            return "t", "U"

        def feature_function2(t, U):
            return "t2", "U2"

        implemented_features = ["nrSpikes", "time_before_first_spike",
                                "spike_rate", "average_AP_overshoot",
                                "average_AHP_depth", "average_AP_width",
                                "accomondation_index"]

        self.uncertainty_calculations.base_features = SpikingFeatures
        self.uncertainty_calculations.features = [feature_function, feature_function2]
        self.assertIsInstance(self.uncertainty_calculations.features, SpikingFeatures)

        t, U = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.uncertainty_calculations.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

        self.assertEqual(set(self.uncertainty_calculations.features.features_to_run),
                         set(["feature_function", "feature_function2"] + implemented_features))


    def test_set_parameters(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error",
                                                                seed=self.seed)
        self.uncertainty_calculations.parameters = Parameters()

        self.assertIsInstance(self.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.parameters, Parameters)


    def test_set_parameter_list(self):
        uncertainty_calculations = UncertaintyCalculations(model=None,
                                                           parameters=None,
                                                           verbose_level="error",
                                                           seed=self.seed)

        uncertainty_calculations.parameters = self.parameterlist

        self.assertIsInstance(uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_error(self):
        uncertainty_calculations = UncertaintyCalculations(model=None,
                                                           parameters=None,
                                                           verbose_level="error",
                                                           seed=self.seed)

        with self.assertRaises(TypeError):
                uncertainty_calculations.parameters = 2

    def test_set_model_function(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error",
                                                                seed=self.seed)

        self.uncertainty_calculations.model = model_function

        self.assertIsInstance(self.uncertainty_calculations.model, Model)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.model, Model)

        self.assertEqual(self.uncertainty_calculations.runmodel.data.xlabel, "")
        self.assertEqual(self.uncertainty_calculations.runmodel.data.ylabel, "")




    def test_create_distribution_none(self):

        self.uncertainty_calculations.create_distribution()

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)

    def test_create_distribution_string(self):

        self.uncertainty_calculations.create_distribution("a")

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)

    def test_create_distribution_list(self):

        self.uncertainty_calculations.create_distribution(["a"])

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)


    def test_create_mask_model(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "TestingModel1d")

        self.assertEqual(len(masked_U), 3)
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(nodes, masked_nodes))


    def test_create_mask_model_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = self.uncertainty_calculations.data.U["TestingModel1d"][0]
        U_2 = self.uncertainty_calculations.data.U["TestingModel1d"][2]

        self.uncertainty_calculations.data.U["TestingModel1d"] = np.array([U_0, np.nan, U_2])

        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "TestingModel1d")


        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


    def test_create_mask_warning(self):
        logfile = os.path.join(self.output_test_dir, "test.log")

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(Distribution(0.5).uniform)

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])


        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="warning",
                                                                verbose_filename=logfile,
                                                                seed=self.seed)

        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = self.uncertainty_calculations.data.U["TestingModel1d"][0]
        U_2 = self.uncertainty_calculations.data.U["TestingModel1d"][2]

        self.uncertainty_calculations.data.U["TestingModel1d"] = np.array([U_0, np.nan, U_2])

        self.uncertainty_calculations.create_mask(nodes, "TestingModel1d")

        print open(logfile).read()
        message = "WARNING - uncertainty_calculations - Feature: TestingModel1d does not yield results for all parameter combinations"
        self.assertTrue(message in open(logfile).read())


    def test_create_mask_feature0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "feature0d")

        self.assertEqual(masked_U[0], 1)
        self.assertEqual(masked_U[1], 1)
        self.assertEqual(masked_U[2], 1)
        self.assertTrue(np.array_equal(nodes, masked_nodes))


    def test_create_mask_feature0d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["feature0d"] = np.array([1, np.nan, 1])
        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "feature0d")

        self.assertTrue(np.array_equal(masked_U, np.array([1, 1])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



    def test_create_mask_feature1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        # feature1d
        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "feature1d")
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10)))
        self.assertTrue(np.array_equal(nodes, masked_nodes))

        # feature2d

    def test_create_mask_feature1d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        self.uncertainty_calculations.data.U["feature1d"] = np.array([np.arange(0, 10), np.nan, np.arange(0, 10)])
        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "feature1d")

        self.assertTrue(np.array_equal(masked_U, np.array([np.arange(0, 10), np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



    def test_create_mask_feature2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "feature2d")
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



    def test_create_mask_feature2d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        U_0 = self.uncertainty_calculations.data.U["feature2d"][0]
        U_2 = self.uncertainty_calculations.data.U["feature2d"][2]

        self.uncertainty_calculations.data.U["feature2d"] = np.array([U_0, np.nan, U_2])

        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "feature2d")

        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_U[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))


    def test_create_mask_feature2d_nodes_1D_nan(self):
        nodes = np.array([0, 1, 2])
        uncertain_parameters = ["a"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        U_0 = self.uncertainty_calculations.data.U["feature2d"][0]
        U_2 = self.uncertainty_calculations.data.U["feature2d"][1]

        self.uncertainty_calculations.data.U["feature2d"] = np.array([U_0, np.nan, U_2])

        masked_nodes, masked_U = self.uncertainty_calculations.create_mask(nodes, "feature2d")

        self.assertTrue(np.array_equal(masked_nodes, np.array([0, 2])))

        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_U[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))



    def test_convert_uncertain_parameters_list(self):
        result = self.uncertainty_calculations.convert_uncertain_parameters(["a", "b"])

        self.assertEqual(result, ["a", "b"])

    def test_convert_uncertain_parameters_string(self):
        result = self.uncertainty_calculations.convert_uncertain_parameters("a")

        self.assertEqual(result, ["a"])

    def test_convert_uncertain_parameters_none(self):
        result = self.uncertainty_calculations.convert_uncertain_parameters(None)

        self.assertEqual(result, ["a", "b"])


    def test_create_PCE_custom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.create_PCE_custom()


    def test_custom_uncertainty_quantification(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.custom_uncertainty_quantification()

    def test_create_PCE_regression_all(self):

        self.uncertainty_calculations.create_PCE_regression()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_regression_one(self):

        self.uncertainty_calculations.create_PCE_regression("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)

    #

    def test_create_PCE_regression_adaptive_error(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(Distribution(1).uniform)

        model = TestingModelAdaptive()
        model.adaptive_model=False

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.uncertainty_calculations = UncertaintyCalculations(model=model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="debug")

        with self.assertRaises(ValueError):
            self.uncertainty_calculations.create_PCE_regression()



    def test_create_PCE_regression_rosenblatt_all(self):

        self.uncertainty_calculations.create_PCE_regression_rosenblatt()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)





    def test_create_PCE_regression_rosenblatt_one(self):

        self.uncertainty_calculations.create_PCE_regression_rosenblatt("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)

    # def test_PCERquadrature(self):
    #
    #     self.uncertainty_calculations.create_PCE_quadrature()
    #
    #     print self.uncertainty_calculations.U_hat
    #     self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
    #     self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)



    def test_calculate_total_sensitivity_1(self):
        self.uncertainty_calculations.data = Data()
        self.uncertainty_calculations.data.sensitivity_1 = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2], "testNone": None}
        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.calculate_total_sensitivity(sensitivity="sensitivity_1")

        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test2D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test2D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test1D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["test1D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_1["testNone"], None)


    def test_calculate_total_sensitivity_t(self):
        self.uncertainty_calculations.data = Data()
        self.uncertainty_calculations.data.sensitivity_t = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2], "testNone": None}
        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.calculate_total_sensitivity(sensitivity="sensitivity_t")

        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test2D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test2D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test1D"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["test1D"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data.total_sensitivity_t["testNone"], None)


    def test_analyse_PCE(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(Distribution(0.5).uniform)

        model = TestingModel1d()


        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")

        self.uncertainty_calculations.data = Data()
        self.uncertainty_calculations.data.feature_list = ["feature0d", "feature1d",
                                                           "feature2d", "TestingModel1d"]

        q0, q1 = cp.variable(2)
        parameter_space = parameters.get_from_uncertain("distribution")
        self.uncertainty_calculations.distribution = cp.J(*parameter_space)

        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]


        self.uncertainty_calculations.U_hat["TestingModel1d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature0d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature1d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature2d"] = cp.Poly([q0, q1*q0, q1])

        self.uncertainty_calculations.analyse_PCE()


        # Test if all calculated properties actually exists
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_05["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.p_95["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_95["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_95["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.p_95["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.E["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.E["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.E["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.E["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.Var["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.Var["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.Var["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.Var["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["TestingModel1d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["feature0d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["feature1d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_1["feature2d"], list)


        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["TestingModel1d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["feature0d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["feature1d"], list)
        self.assertIsInstance(self.uncertainty_calculations.data.total_sensitivity_t["feature2d"], list)



        self.assertIsInstance(self.uncertainty_calculations.U_mc["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature2d"], np.ndarray)


    def test_polynomial_chaos(self):
        data = self.uncertainty_calculations.polynomial_chaos()

        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_PC_rosenblatt(self):
        data = self.uncertainty_calculations.polynomial_chaos(rosenblatt=True)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_Rosenblatt.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_Rosenblatt.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_PC_custom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.polynomial_chaos(method="custom")



    def test_PC_error(self):
        with self.assertRaises(ValueError):
            self.uncertainty_calculations.polynomial_chaos(method="not implemented")


    def test_PC_parameter_a(self):
        data = self.uncertainty_calculations.polynomial_chaos("a")

        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-a.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC_parameter_b(self):
        data = self.uncertainty_calculations.polynomial_chaos("b")

        filename = os.path.join(self.output_test_dir, "UncertaintyCalculations_single-parameter-b.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/UncertaintyCalculations_single-parameter-b.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_monte_carlo(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(Distribution(0.5).uniform)

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")



        data = self.uncertainty_calculations.monte_carlo()

        # Rough tests
        self.assertTrue(np.allclose(self.uncertainty_calculations.data.E["TestingModel1d"],
                                    np.arange(0, 10) + 3, atol=0.1))
        self.assertTrue(np.allclose(self.uncertainty_calculations.data.Var["TestingModel1d"],
                                    np.zeros(10), atol=0.1))
        self.assertTrue(np.all(np.less(self.uncertainty_calculations.data.p_05["TestingModel1d"],
                                       np.arange(0, 10) + 3)))
        self.assertTrue(np.all(np.greater(self.uncertainty_calculations.data.p_95["TestingModel1d"],
                                          np.arange(0, 10) + 3)))


        # Compare to pregenerated data
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)




    def test_monte_carlo_feature0d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(Distribution(0.5).uniform)

        model = TestingModel1d()
        features = TestingFeatures(features_to_run=["feature0d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")



        self.uncertainty_calculations.monte_carlo()

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.E["feature0d"],
                                       features.feature0d(None, None)[1]))
        self.assertEqual(self.uncertainty_calculations.data.Var["feature0d"], 0)
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_05["feature0d"],
                                       features.feature0d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_95["feature0d"],
                                       features.feature0d(None, None)[1]))


    def test_monte_carlo_feature1d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(Distribution(0.5).uniform)

        model = TestingModel1d()
        features = TestingFeatures(features_to_run=["feature1d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")



        self.uncertainty_calculations.monte_carlo()

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.E["feature1d"],
                                       features.feature1d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.Var["feature1d"],
                                       np.zeros(10)))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_05["feature1d"],
                                       features.feature1d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_95["feature1d"],
                                       features.feature1d(None, None)[1]))



    def test_monte_carlo_feature2d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(Distribution(0.5).uniform)

        model = TestingModel1d()
        features = TestingFeatures(features_to_run=["feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                nr_mc_samples=10**1,
                                                                verbose_level="error")


        self.uncertainty_calculations.monte_carlo()

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.E["feature2d"],
                                       features.feature2d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.Var["feature2d"],
                                       np.zeros((2, 10))))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_05["feature2d"],
                                       features.feature2d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data.p_95["feature2d"],
                                       features.feature2d(None, None)[1]))
