import unittest
import os
import shutil
import subprocess

import numpy as np
import chaospy as cp

from uncertainpy.core import UncertaintyCalculations
from uncertainpy.parameters import Parameters
from uncertainpy.features import Features
from uncertainpy import uniform, normal
from uncertainpy import Data
from uncertainpy.models import Model
from uncertainpy import SpikingFeatures


from .testing_classes import TestingFeatures
from .testing_classes import TestingModel1d, model_function
from .testing_classes import TestingModelAdaptive, TestingModelIncomplete


class TestUncertaintyCalculations(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10
        self.difference_threshold = 1e-10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.parameter_list = [["a", 1, None],
                               ["b", 2, None]]

        self.parameters = Parameters(self.parameter_list)
        self.parameters.set_all_distributions(uniform(0.5))

        self.model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                                parameters=self.parameters,
                                                                features=features,
                                                                verbose_level="error")

    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        uncertainty = UncertaintyCalculations(model=self.model,
                                              parameters=self.parameters)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.features, Features)



    def test_intit_features(self):
        uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                           verbose_level="error")
        self.assertIsInstance(uncertainty_calculations.features, Features)

        uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                           parameters=None,
                                                           features=TestingFeatures(),
                                                           verbose_level="error")
        self.assertIsInstance(uncertainty_calculations.features, TestingFeatures)


    def test_set_model(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error")


        self.uncertainty_calculations.model = self.model

        self.assertIsInstance(self.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.model, TestingModel1d)


    def test_set_features(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error")
        self.uncertainty_calculations.features = TestingFeatures()

        self.assertIsInstance(self.uncertainty_calculations.features, TestingFeatures)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.features, TestingFeatures)


    def test_feature_function(self):
        def feature_function(time, values):
                return "time", "values"

        self.uncertainty_calculations.features = feature_function
        self.assertIsInstance(self.uncertainty_calculations.features, Features)

        time, values = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        self.assertEqual(self.uncertainty_calculations.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(time, values):
                return "time", "values"

        def feature_function2(time, values):
                return "t2", "U2"


        self.uncertainty_calculations.features = [feature_function, feature_function2]
        self.assertIsInstance(self.uncertainty_calculations.features, Features)

        time, values = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")


        time, values = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        time, values = self.uncertainty_calculations.features.feature_function2(None, None)
        self.assertEqual(time, "t2")
        self.assertEqual(values, "U2")

        self.assertEqual(self.uncertainty_calculations.features.features_to_run,
                         ["feature_function", "feature_function2"])


    def test_feature_functions_base(self):
        def feature_function(time, values):
            return "time", "values"

        def feature_function2(time, values):
            return "time2", "values2"

        implemented_features = ["nr_spikes", "time_before_first_spike",
                                "spike_rate", "average_AP_overshoot",
                                "average_AHP_depth", "average_AP_width",
                                "accommodation_index"]

        self.uncertainty_calculations.features = SpikingFeatures([feature_function, feature_function2])
        self.assertIsInstance(self.uncertainty_calculations.features, SpikingFeatures)

        time, values = self.uncertainty_calculations.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        time, values = self.uncertainty_calculations.features.feature_function2(None, None)
        self.assertEqual(time, "time2")
        self.assertEqual(values, "values2")

        self.assertEqual(set(self.uncertainty_calculations.features.features_to_run),
                         set(["feature_function", "feature_function2"] + implemented_features))


    def test_set_parameters(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error")
        self.uncertainty_calculations.parameters = Parameters()

        self.assertIsInstance(self.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.parameters, Parameters)


    def test_set_parameter_list(self):
        uncertainty_calculations = UncertaintyCalculations(model=None,
                                                           parameters=None,
                                                           verbose_level="error")

        uncertainty_calculations.parameters = self.parameter_list

        self.assertIsInstance(uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_error(self):
        uncertainty_calculations = UncertaintyCalculations(model=None,
                                                           parameters=None,
                                                           verbose_level="error")

        with self.assertRaises(TypeError):
            uncertainty_calculations.parameters = 2


    def test_set_model_function(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                verbose_level="error")

        self.uncertainty_calculations.model = model_function

        self.assertIsInstance(self.uncertainty_calculations.model, Model)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.model, Model)


    def test_create_distribution_dist(self):

        dist = cp.J(cp.Uniform())
        self.uncertainty_calculations.parameters.distribution = dist
        distribution = self.uncertainty_calculations.create_distribution()

        self.assertIsInstance(distribution, cp.Dist)
        self.assertEqual(distribution, dist)

    def test_create_distribution_none(self):

        distribution = self.uncertainty_calculations.create_distribution()

        self.assertIsInstance(distribution, cp.Dist)

    def test_create_distribution_string(self):

        distribution = self.uncertainty_calculations.create_distribution("a")

        self.assertIsInstance(distribution, cp.Dist)

    def test_create_distribution_list(self):

        distribution = self.uncertainty_calculations.create_distribution(["a"])

        self.assertIsInstance(distribution, cp.Dist)


    def test_create_mask_model(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "TestingModel1d")

        self.assertEqual(len(masked_values), 3)
        self.assertTrue(np.array_equal(masked_values[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_values[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_values[2], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(nodes, masked_nodes))
        self.assertTrue(np.all(mask))



    def test_create_mask_model_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = self.uncertainty_calculations.data["TestingModel1d"]["values"][0]
        U_2 = self.uncertainty_calculations.data["TestingModel1d"]["values"][2]

        self.uncertainty_calculations.data["TestingModel1d"]["values"] = np.array([U_0, np.nan, U_2])

        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "TestingModel1d")


        self.assertEqual(len(masked_values), 2)
        self.assertTrue(np.array_equal(masked_values[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_values[1], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))


    def test_create_mask_warning(self):
        logfile = os.path.join(self.output_test_dir, "test.log")

        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])


        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="warning",
                                                                verbose_filename=logfile)

        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = self.uncertainty_calculations.data["TestingModel1d"]["values"][0]
        U_2 = self.uncertainty_calculations.data["TestingModel1d"]["values"][2]

        self.uncertainty_calculations.data["TestingModel1d"]["values"] = np.array([U_0, np.nan, U_2])

        self.uncertainty_calculations.create_mask(nodes, "TestingModel1d")

        message = "WARNING - Feature: TestingModel1d only yields results for 2/3 parameter combinations"
        self.assertTrue(message in open(logfile).read())


    def test_create_mask_feature0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "feature0d")

        self.assertEqual(masked_values[0], 1)
        self.assertEqual(masked_values[1], 1)
        self.assertEqual(masked_values[2], 1)
        self.assertTrue(np.array_equal(nodes, masked_nodes))
        self.assertTrue(np.all(mask))



    def test_create_mask_feature0d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data["feature0d"]["values"] = np.array([1, np.nan, 1])
        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "feature0d")

        self.assertTrue(np.array_equal(masked_values, np.array([1, 1])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))



    def test_create_mask_feature1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        # feature1d
        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "feature1d")
        self.assertTrue(np.array_equal(masked_values[0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_values[1], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_values[2], np.arange(0, 10)))
        self.assertTrue(np.array_equal(nodes, masked_nodes))
        self.assertTrue(np.all(mask))

        # feature2d

    def test_create_mask_feature1d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        self.uncertainty_calculations.data["feature1d"]["values"] = np.array([np.arange(0, 10), np.nan, np.arange(0, 10)])
        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "feature1d")

        self.assertTrue(np.array_equal(masked_values, np.array([np.arange(0, 10), np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))



    def test_create_mask_feature2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "feature2d")
        self.assertTrue(np.array_equal(masked_values[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_values[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_values[2],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(nodes, masked_nodes))
        self.assertTrue(np.all(mask))


    def test_create_mask_feature2d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        U_0 = self.uncertainty_calculations.data["feature2d"]["values"][0]
        U_2 = self.uncertainty_calculations.data["feature2d"]["values"][2]
        self.uncertainty_calculations.data["feature2d"]["values"] = np.array([U_0, np.nan, U_2])

        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "feature2d")

        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


        self.assertEqual(len(masked_values), 2)
        self.assertTrue(np.array_equal(masked_values[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_values[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))

        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))


    def test_create_mask_feature2d_nodes_1D_nan(self):
        nodes = np.array([0, 1, 2])
        uncertain_parameters = ["a"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        U_0 = self.uncertainty_calculations.data["feature2d"]["values"][0]
        U_2 = self.uncertainty_calculations.data["feature2d"]["values"][1]

        self.uncertainty_calculations.data["feature2d"]["values"] = np.array([U_0, np.nan, U_2])

        masked_nodes, masked_values, mask = self.uncertainty_calculations.create_mask(nodes, "feature2d")

        self.assertTrue(np.array_equal(masked_nodes, np.array([0, 2])))

        self.assertEqual(len(masked_values), 2)
        self.assertTrue(np.array_equal(masked_values[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_values[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))

        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))


    def test_convert_uncertain_parameters_list(self):
        result = self.uncertainty_calculations.convert_uncertain_parameters(["a", "b"])

        self.assertEqual(result, ["a", "b"])

    def test_convert_uncertain_parameters_string(self):
        result = self.uncertainty_calculations.convert_uncertain_parameters("a")

        self.assertEqual(result, ["a"])

    def test_convert_uncertain_parameters_none(self):
        result = self.uncertainty_calculations.convert_uncertain_parameters(None)

        self.assertEqual(result, ["a", "b"])


    def test_convert_uncertain_parameters_error(self):
        self.parameter_list = [["a", 1, None],
                               ["b", 2, None]]

        self.parameters = Parameters(self.parameter_list)

        self.parameters.distribution = cp.J(cp.Uniform(), cp.Uniform())

        self.model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                                parameters=self.parameters,
                                                                features=features,
                                                                verbose_level="error")

        with self.assertRaises(ValueError):
            result = self.uncertainty_calculations.convert_uncertain_parameters(["a"])


    def test_convert_uncertain_parameters_distribution(self):
        self.parameter_list = [["a", 1, None],
                               ["b", 2, None]]

        self.parameters = Parameters(self.parameter_list)

        self.parameters.distribution = cp.J(cp.Uniform(), cp.Uniform())

        self.model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                                parameters=self.parameters,
                                                                features=features,
                                                                verbose_level="error")


        result = self.uncertainty_calculations.convert_uncertain_parameters(None)
        self.assertEqual(result, ["a", "b"])



    def test_convert_uncertain_parameters_distribution_list(self):
        self.parameter_list = [["a", 1, None],
                               ["b", 2, None]]

        self.parameters = Parameters(self.parameter_list)

        self.parameters.distribution = cp.J(cp.Uniform(), cp.Uniform())

        self.model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                                parameters=self.parameters,
                                                                features=features,
                                                                verbose_level="error")


        result = self.uncertainty_calculations.convert_uncertain_parameters(["a", "b"])
        self.assertEqual(result, ["a", "b"])


    def test_create_PCE_custom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.create_PCE_custom()


    def test_custom_uncertainty_quantification(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.custom_uncertainty_quantification()


    def test_create_PCE_collocation_all(self):
        self.uncertainty_calculations.create_PCE_collocation()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_collocation_all_parameters(self):
        self.uncertainty_calculations.create_PCE_collocation(polynomial_order=5,
                                                             nr_collocation_nodes=22)

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_collocation_one(self):

        self.uncertainty_calculations.create_PCE_collocation("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)

    #

    def test_create_PCE_collocation_adaptive_error(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(1))

        model = TestingModelAdaptive()
        model.adaptive=False

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.uncertainty_calculations = UncertaintyCalculations(model=model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="debug")

        with self.assertRaises(ValueError):
            self.uncertainty_calculations.create_PCE_collocation()



    def test_create_PCE_collocation_rosenblatt_all(self):

        self.uncertainty_calculations.create_PCE_collocation_rosenblatt()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_collocation_rosenblatt_one(self):

        self.uncertainty_calculations.create_PCE_collocation_rosenblatt("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_spectral_all(self):
        self.uncertainty_calculations.create_PCE_spectral()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_spectral_all_parameters(self):
        self.uncertainty_calculations.create_PCE_spectral(polynomial_order=4,
                                                          quadrature_order=6)

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_spectral_one(self):

        self.uncertainty_calculations.create_PCE_spectral("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_collocation_adaptive_error(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(1))

        model = TestingModelAdaptive()
        model.adaptive=False

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.uncertainty_calculations = UncertaintyCalculations(model=model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="debug")

        with self.assertRaises(ValueError):
            self.uncertainty_calculations.create_PCE_spectral()



    def test_create_PCE_spectral_rosenblatt_all(self):

        self.uncertainty_calculations.create_PCE_spectral_rosenblatt()

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_spectral_rosenblatt_one(self):

        self.uncertainty_calculations.create_PCE_spectral_rosenblatt("a")

        self.assertEqual(self.uncertainty_calculations.data.uncertain_parameters, ["a"])
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty_calculations.U_hat["TestingModel1d"], cp.Poly)



    def test_calculate_sensitivity_1_sum(self):
        self.uncertainty_calculations.data = Data()

        self.uncertainty_calculations.data.add_features(["test2D", "test1D"])
        self.uncertainty_calculations.data["test2D"].sensitivity_1 = [[4, 6], [8, 12]]
        self.uncertainty_calculations.data["test1D"].sensitivity_1 =  [1, 2]
        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.calculate_sensitivity_sum(sensitivity="sensitivity_1")

        self.assertEqual(self.uncertainty_calculations.data["test2D"]["sensitivity_1_sum"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data["test2D"]["sensitivity_1_sum"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data["test1D"]["sensitivity_1_sum"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data["test1D"]["sensitivity_1_sum"][1], 2/3.)


    def test_calculate_sensitivity_t_sum(self):
        self.uncertainty_calculations.data = Data()

        self.uncertainty_calculations.data.add_features(["test2D", "test1D"])
        self.uncertainty_calculations.data["test2D"].sensitivity_t = [[4, 6], [8, 12]]
        self.uncertainty_calculations.data["test1D"].sensitivity_t = [1, 2]
        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.calculate_sensitivity_sum(sensitivity="sensitivity_t")

        self.assertEqual(self.uncertainty_calculations.data["test2D"]["sensitivity_t_sum"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data["test2D"]["sensitivity_t_sum"][1], 2/3.)
        self.assertEqual(self.uncertainty_calculations.data["test1D"]["sensitivity_t_sum"][0], 1/3.)
        self.assertEqual(self.uncertainty_calculations.data["test1D"]["sensitivity_t_sum"][1], 2/3.)



    def test_analyse_PCE(self):
        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()


        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")

        self.uncertainty_calculations.data = Data()

        q0, q1 = cp.variable(2)
        parameter_space = parameters.get_from_uncertain("distribution")
        self.uncertainty_calculations.distribution = cp.J(*parameter_space)

        self.uncertainty_calculations.data.uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data.add_features(["TestingModel1d", "feature0d",
                                                        "feature1d", "feature2d"])

        self.uncertainty_calculations.U_hat["TestingModel1d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature0d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature1d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty_calculations.U_hat["feature2d"] = cp.Poly([q0, q1*q0, q1])

        self.uncertainty_calculations.analyse_PCE()

        # Test if all calculated properties actually exists
        data = self.uncertainty_calculations.data

        data_types = ["values", "time", "mean", "variance", "percentile_5", "percentile_95",
                      "sensitivity_1", "sensitivity_1_sum",
                      "sensitivity_t", "sensitivity_t_sum", "labels"]

        for data_type in data_types:
            if data_type not in ["values", "time", "labels"]:
                for feature in data:
                    self.assertIsInstance(data[feature][data_type], np.ndarray)




        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_5["TestingModel1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_5["feature0d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_5["feature1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_5["feature2d"], np.ndarray)

        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_95["TestingModel1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_95["feature0d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_95["feature1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.percentile_95["feature2d"], np.ndarray)

        # self.assertIsInstance(self.uncertainty_calculations.data.E["TestingModel1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.E["feature0d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.E["feature1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.E["feature2d"], np.ndarray)

        # self.assertIsInstance(self.uncertainty_calculations.data.variance["TestingModel1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.variance["feature0d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.variance["feature1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.variance["feature2d"], np.ndarray)

        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["TestingModel1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature0d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1["feature2d"], np.ndarray)

        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1_sum["TestingModel1d"], list)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1_sum["feature0d"], list)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1_sum["feature1d"], list)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_1_sum["feature2d"], list)

        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["TestingModel1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature0d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature1d"], np.ndarray)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t["feature2d"], np.ndarray)

        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t_sum["TestingModel1d"], list)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t_sum["feature0d"], list)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t_sum["feature1d"], list)
        # self.assertIsInstance(self.uncertainty_calculations.data.sensitivity_t_sum["feature2d"], list)

        self.assertIsInstance(self.uncertainty_calculations.U_mc["TestingModel1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty_calculations.U_mc["feature2d"], np.ndarray)


    def test_polynomial_chaos_collocation(self):
        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_threshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_PC_collocation_rosenblatt(self):
        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt=True,
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_Rosenblatt.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_Rosenblatt.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_threshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_polynomial_chaos_spectral(self):
        data = self.uncertainty_calculations.polynomial_chaos(method="spectral",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_spectral.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_spectral.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_threshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_PC_collocation_spectral(self):
        data = self.uncertainty_calculations.polynomial_chaos(method="spectral",
                                                              rosenblatt=True,
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_Rosenblatt_spectral.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_Rosenblatt_spectral.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_threshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_PC_custom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.polynomial_chaos(method="custom",
                                                           seed=self.seed)



    def test_PC_error(self):
        with self.assertRaises(ValueError):
            self.uncertainty_calculations.polynomial_chaos(method="not implemented",
                                                           seed=self.seed)


    def test_PC_parameter_a(self):
        data = self.uncertainty_calculations.polynomial_chaos("a",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-a.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_threshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC_parameter_b(self):
        data = self.uncertainty_calculations.polynomial_chaos("b",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "UncertaintyCalculations_single-parameter-b.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/UncertaintyCalculations_single-parameter-b.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_threshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)



    def test_monte_carlo(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")


        data = self.uncertainty_calculations.monte_carlo(nr_samples=10**1)

        # Rough tests
        self.assertTrue(np.allclose(self.uncertainty_calculations.data["TestingModel1d"]["mean"],
                                    np.arange(0, 10) + 3, atol=0.1))
        self.assertTrue(np.allclose(self.uncertainty_calculations.data["TestingModel1d"]["variance"],
                                    np.zeros(10), atol=0.1))
        self.assertTrue(np.all(np.less(self.uncertainty_calculations.data["TestingModel1d"]["percentile_5"],
                                       np.arange(0, 10) + 3)))
        self.assertTrue(np.all(np.greater(self.uncertainty_calculations.data["TestingModel1d"]["percentile_95"],
                                          np.arange(0, 10) + 3)))


        # Compare to pregenerated data
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_threshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)




    def test_monte_carlo_feature0d(self):
        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()
        features = TestingFeatures(features_to_run=["feature0d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")



        self.uncertainty_calculations.monte_carlo(nr_samples=10**1)

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature0d"]["mean"],
                                       features.feature0d(None, None)[1]))
        self.assertEqual(self.uncertainty_calculations.data["feature0d"]["variance"], 0)
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature0d"]["percentile_5"],
                                       features.feature0d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature0d"]["percentile_95"],
                                       features.feature0d(None, None)[1]))


    def test_monte_carlo_feature1d(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()
        features = TestingFeatures(features_to_run=["feature1d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")



        self.uncertainty_calculations.monte_carlo(nr_samples=10**1)

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature1d"]["mean"],
                                       features.feature1d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature1d"]["variance"],
                                       np.zeros(10)))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature1d"]["percentile_5"],
                                       features.feature1d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature1d"]["percentile_95"],
                                       features.feature1d(None, None)[1]))



    def test_monte_carlo_feature2d(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()
        features = TestingFeatures(features_to_run=["feature2d"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")


        self.uncertainty_calculations.monte_carlo(nr_samples=10**1)

        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature2d"]["mean"],
                                       features.feature2d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature2d"]["variance"],
                                       np.zeros((2, 10))))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature2d"]["percentile_5"],
                                       features.feature2d(None, None)[1]))
        self.assertTrue(np.array_equal(self.uncertainty_calculations.data["feature2d"]["percentile_95"],
                                       features.feature2d(None, None)[1]))


    def test_create_PCE_collocation_incomplete(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModelIncomplete()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error",
                                                                allow_incomplete=False)

        self.uncertainty_calculations.create_PCE_collocation(["a", "b"],
                                                             seed=self.seed)

        self.assertEqual(self.uncertainty_calculations.U_hat, {})
        self.assertEqual(self.uncertainty_calculations.data.incomplete,
                         ["TestingModelIncomplete"])





    def test_create_PCE_collocation_incomplete(self):
        np.random.seed(self.seed)

        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModelIncomplete()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")

        self.uncertainty_calculations.create_PCE_collocation(["a", "b"],
                                                             allow_incomplete=False)

        self.assertEqual(self.uncertainty_calculations.U_hat, {})
        self.assertEqual(self.uncertainty_calculations.data.incomplete,
                         ["TestingModelIncomplete"])



        # TODO Find a good way to test this case when not all runs contains None
        # self.uncertainty_calculations = UncertaintyCalculations(model,
        #                                                         parameters=parameters,
        #                                                         features=features,
        #                                                         verbose_level="error",
        #                                                         seed=self.seed,
        #                                                         allow_incomplete=True)

        # self.uncertainty_calculations.create_PCE_collocation(["a", "b"])

        # self.assertIn("TestingModelIncomplete", self.uncertainty_calculations.U_hat)
        # self.assertEqual(self.uncertainty_calculations.data.incomplete,
        #                  ["TestingModelIncomplete"])




    def test_create_PCE_spectral_rosenblatt_incomplete(self):
        np.random.seed(self.seed)

        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModelIncomplete()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")

        self.uncertainty_calculations.create_PCE_spectral(["a", "b"],
                                                             allow_incomplete=False)

        self.assertEqual(self.uncertainty_calculations.U_hat, {})
        self.assertEqual(self.uncertainty_calculations.data.incomplete,
                         ["TestingModelIncomplete"])

        # TODO Find a good way to test this case when not all runs contains None
        # self.uncertainty_calculations = UncertaintyCalculations(model,
        #                                                         parameters=parameters,
        #                                                         features=features,
        #                                                         verbose_level="error",
        #                                                         seed=self.seed,
        #                                                         allow_incomplete=True)

        # self.uncertainty_calculations.create_PCE_spectral_rosenblatt(["a", "b"])

        # self.assertIn("TestingModelIncomplete", self.uncertainty_calculations.U_hat)
        # self.assertEqual(self.uncertainty_calculations.data.incomplete,
        #                  ["TestingModelIncomplete"])



    def test_create_PCE_spectral_incomplete(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModelIncomplete()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error",
                                                                allow_incomplete=False)

        self.uncertainty_calculations.create_PCE_spectral(["a", "b"],
                                                           seed=self.seed)

        self.assertEqual(self.uncertainty_calculations.U_hat, {})
        self.assertEqual(self.uncertainty_calculations.data.incomplete,
                         ["TestingModelIncomplete"])


    def test_create_PCE_spectral_incomplete(self):
        np.random.seed(self.seed)

        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModelIncomplete()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")

        self.uncertainty_calculations.create_PCE_spectral(["a", "b"],
                                                          allow_incomplete=False)

        self.assertEqual(self.uncertainty_calculations.U_hat, {})
        self.assertEqual(self.uncertainty_calculations.data.incomplete,
                         ["TestingModelIncomplete"])


    def test_create_PCE_spectral_rosenblatt_incomplete(self):
        np.random.seed(self.seed)

        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModelIncomplete()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                verbose_level="error")

        self.uncertainty_calculations.create_PCE_spectral(["a", "b"],
                                                           allow_incomplete=False)

        self.assertEqual(self.uncertainty_calculations.U_hat, {})
        self.assertEqual(self.uncertainty_calculations.data.incomplete,
                         ["TestingModelIncomplete"])
