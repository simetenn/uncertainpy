import sys
import unittest
import os
import shutil
import subprocess
import time
import logging
import numpy as np
import chaospy as cp

from uncertainpy.core import UncertaintyCalculations
from uncertainpy.parameters import Parameters
from uncertainpy.features import Features
from uncertainpy import uniform, normal
from uncertainpy import Data
from uncertainpy.models import Model
from uncertainpy import SpikingFeatures

from SALib.analyze.sobol import separate_output_values


from .testing_classes import TestingFeatures
from .testing_classes import TestingModel1d, model_function
from .testing_classes import TestingModelAdaptive, TestingModelIncomplete

from uncertainpy.utils.logger import add_file_handler

class TestUncertaintyCalculations(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.threshold = 1e-10
        self.seed = 10
        self.nr_mc_samples = 10

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
                                                                logger_level="error")

    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                              parameters=self.parameters,
                                              logger_level="error")

        self.assertIsInstance(uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(uncertainty_calculations.features, Features)



    def test_intit_features(self):
        uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                           logger_level="error")
        self.assertIsInstance(uncertainty_calculations.features, Features)

        uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                           parameters=None,
                                                           features=TestingFeatures(),
                                                           logger_level="error")

        self.assertIsInstance(uncertainty_calculations.features, TestingFeatures)


    def test_set_model(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                logger_level="error")


        self.uncertainty_calculations.model = self.model

        self.assertIsInstance(self.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.model, TestingModel1d)


    def test_set_features(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                logger_level="error")
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
                                                                logger_level="error")
        self.uncertainty_calculations.parameters = Parameters()

        self.assertIsInstance(self.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.parameters, Parameters)


    def test_set_parameter_list(self):
        uncertainty_calculations = UncertaintyCalculations(model=None,
                                                           parameters=None,
                                                           logger_level="error")

        uncertainty_calculations.parameters = self.parameter_list

        self.assertIsInstance(uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_error(self):
        uncertainty_calculations = UncertaintyCalculations(model=None,
                                                           parameters=None,
                                                           logger_level="error")

        with self.assertRaises(TypeError):
            uncertainty_calculations.parameters = 2


    def test_set_model_function(self):
        self.uncertainty_calculations = UncertaintyCalculations(model=None,
                                                                parameters=None,
                                                                logger_level="error")

        self.uncertainty_calculations.model = model_function

        self.assertIsInstance(self.uncertainty_calculations.model, Model)
        self.assertIsInstance(self.uncertainty_calculations.runmodel.model, Model)


    def test_create_distribution_dist(self):

        dist = cp.J(cp.Uniform(), cp.Uniform())
        self.uncertainty_calculations.parameters.distribution = dist
        distribution = self.uncertainty_calculations.create_distribution()

        self.assertIsInstance(distribution, cp.Dist)
        self.assertEqual(distribution, dist)

    def test_create_distribution_dist_error(self):

        dist = cp.J(cp.Uniform(), cp.Uniform())
        self.uncertainty_calculations.parameters.distribution = dist

        with self.assertRaises(ValueError):
            self.uncertainty_calculations.create_distribution("a")


    def test_create_distribution_none(self):

        distribution = self.uncertainty_calculations.create_distribution()

        self.assertIsInstance(distribution, cp.Dist)

    def test_create_distribution_string(self):

        distribution = self.uncertainty_calculations.create_distribution("a")

        self.assertIsInstance(distribution, cp.Dist)

    def test_create_distribution_list(self):

        distribution = self.uncertainty_calculations.create_distribution(["a"])

        self.assertIsInstance(distribution, cp.Dist)



    def test_create_mask(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_evaluations, mask = \
            self.uncertainty_calculations.create_mask(data["TestingModel1d"].evaluations)

        self.assertEqual(len(masked_evaluations), 3)
        self.assertTrue(np.array_equal(masked_evaluations[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_evaluations[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_evaluations[2], np.arange(0, 10) + 5))
        self.assertTrue(np.all(mask))


    def test_create_masked_evaluations(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_evaluations, mask = \
            self.uncertainty_calculations.create_masked_evaluations(data, "TestingModel1d")

        self.assertEqual(len(masked_evaluations), 3)
        self.assertTrue(np.array_equal(masked_evaluations[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_evaluations[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_evaluations[2], np.arange(0, 10) + 5))
        self.assertTrue(np.all(mask))


    def test_create_masked_nodes(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_evaluations, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "TestingModel1d", nodes)

        self.assertEqual(len(masked_evaluations), 3)
        self.assertTrue(np.array_equal(masked_evaluations[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_evaluations[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_evaluations[2], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(np.array([[0, 1, 2], [1, 2, 3]]), masked_nodes))
        self.assertTrue(np.all(mask))



    def test_create_masked_nodes_weights(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]
        weights = np.array([0, 1, 2])

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_evaluations, mask, masked_nodes, masked_weights = \
            self.uncertainty_calculations.create_masked_nodes_weights(data, "TestingModel1d", nodes, weights)

        self.assertEqual(len(masked_evaluations), 3)
        self.assertTrue(np.array_equal(masked_evaluations[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_evaluations[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_evaluations[2], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(np.array([[0, 1, 2], [1, 2, 3]]), masked_nodes))
        self.assertTrue(np.array_equal(np.array([0, 1, 2]), masked_weights))
        self.assertTrue(np.all(mask))



    def test_create_masked_evaluations_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = data["TestingModel1d"].evaluations[0]
        U_2 = data["TestingModel1d"].evaluations[2]

        data["TestingModel1d"].evaluations = [U_0, np.nan, U_2]


        masked_evaluations, mask = \
            self.uncertainty_calculations.create_masked_evaluations(data, "TestingModel1d")

        self.assertEqual(len(masked_evaluations), 2)
        self.assertTrue(np.array_equal(masked_evaluations[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_evaluations[1], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))


    def test_create_masked_nodes_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = data["TestingModel1d"].evaluations[0]
        U_2 = data["TestingModel1d"].evaluations[2]

        data["TestingModel1d"].evaluations = [U_0, np.nan, U_2]


        masked_evaluations, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "TestingModel1d", nodes)

        self.assertEqual(len(masked_evaluations), 2)
        self.assertTrue(np.array_equal(masked_evaluations[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_evaluations[1], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))




    def test_create_masked_evaluations_nested_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = data["TestingModel1d"].evaluations[0]
        U_2 = data["TestingModel1d"].evaluations[2]

        data["TestingModel1d"].evaluations = [U_0, [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10], U_2]

        masked_evaluations, mask = \
            self.uncertainty_calculations.create_masked_evaluations(data, "TestingModel1d")

        self.assertEqual(len(masked_evaluations), 2)
        self.assertTrue(np.array_equal(masked_evaluations[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_evaluations[1], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))



    def test_create_masked_evaluations_warning(self):
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
                                                                logger_level="warning")


        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        # add handler to examine if we print the warning
        add_file_handler("uncertainpy", filename=logfile)

        U_0 = data["TestingModel1d"].evaluations[0]
        U_2 = data["TestingModel1d"].evaluations[2]

        data["TestingModel1d"].evaluations = [U_0, np.nan, U_2]

        self.uncertainty_calculations.create_masked_evaluations(data, "TestingModel1d")

        time.sleep(0.4)
        message = "WARNING - TestingModel1d: only yields results for 2/3 parameter combinations."

        self.assertTrue(message in open(logfile).read())

        # remove the handler we have added
        logger = logging.getLogger("uncertainpy")

        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)








    def test_create_masked_nodes_feature0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_values, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "feature0d", nodes)

        self.assertEqual(masked_values[0], 1)
        self.assertEqual(masked_values[1], 1)
        self.assertEqual(masked_values[2], 1)
        self.assertTrue(np.array_equal(nodes, masked_nodes))
        self.assertTrue(np.all(mask))



    def test_create_masked_nodes_feature0d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        data["feature0d"].evaluations = [1, np.nan, 1]

        masked_values, mask, masked_nodes = \
        self.uncertainty_calculations.create_masked_nodes(data, "feature0d", nodes)

        self.assertTrue(np.array_equal(masked_values, np.array([1, 1])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))



    def test_create_masked_nodes_feature1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        # feature1d
        masked_values, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "feature1d", nodes)

        self.assertTrue(np.array_equal(masked_values[0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_values[1], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_values[2], np.arange(0, 10)))
        self.assertTrue(np.array_equal(nodes, masked_nodes))
        self.assertTrue(np.all(mask))


    def test_create_masked_nodes_feature1d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        data["feature1d"].evaluations = [np.arange(0, 10), np.nan, np.arange(0, 10)]

        masked_values, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "feature1d", nodes)

        self.assertTrue(np.array_equal(masked_values, np.array([np.arange(0, 10), np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))


    def test_create_masked_nodes_feature1d_nested_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        data["feature1d"].evaluations = [np.arange(0, 10), [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10], np.arange(0, 10)]

        masked_values, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "feature1d", nodes)

        self.assertTrue(np.array_equal(masked_values, np.array([np.arange(0, 10), np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))


    def test_create_masked_nodes_feature2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_values, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "feature2d", nodes)

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


    def test_create_masked_nodes_feature2d_nan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        U_0 = data["feature2d"].evaluations[0]
        U_2 = data["feature2d"].evaluations[2]
        data["feature2d"].evaluations = [U_0, np.nan, U_2]

        masked_values, mask, masked_nodes = \
            self.uncertainty_calculations.create_masked_nodes(data, "feature2d", nodes)

        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))

        self.assertEqual(len(masked_values), 2)
        self.assertTrue(np.array_equal(masked_values[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_values[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))

        self.assertTrue(np.array_equal(mask, np.array([True, False, True])))


    def test_create_masked_nodes_feature2d_nodes_1D_nan(self):
        nodes = np.array([0, 1, 2])
        uncertain_parameters = ["a"]

        data = self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        U_0 = data["feature2d"].evaluations[0]
        U_2 = data["feature2d"].evaluations[1]

        data["feature2d"].evaluations = [U_0, np.nan, U_2]

        masked_values, mask, masked_nodes =\
            self.uncertainty_calculations.create_masked_nodes(data, "feature2d", nodes)

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
                                                                logger_level="error")

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
                                                                logger_level="error")


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
                                                                logger_level="error")


        result = self.uncertainty_calculations.convert_uncertain_parameters(["a", "b"])
        self.assertEqual(result, ["a", "b"])


    def test_create_PCE_custom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.create_PCE_custom()


    def test_create_PCE_custom_assign(self):
        def create_PCE_custom(self, uncertain_parameters=None, custom=None):
            return "Custom pce", custom

        self.uncertainty_calculations.create_PCE_custom = create_PCE_custom

        data, custom = self.uncertainty_calculations.create_PCE_custom(custom="test")
        self.assertEqual(data, "Custom pce")
        self.assertEqual(custom, "test")


    def test_create_PCE_custom_assign_no_arguments(self):
        def create_PCE_custom():
            return "Custom pce"

        self.uncertainty_calculations.create_PCE_custom = create_PCE_custom

        with self.assertRaises(TypeError):
            self.uncertainty_calculations.create_PCE_custom()


    def test_custom_uncertainty_quantification(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.custom_uncertainty_quantification()



    def test_custom_uncertainty_quantification_assigned(self):
        def custom_method(self, argument=None):
            return "custom"

        self.uncertainty_calculations.custom_uncertainty_quantification = custom_method
        result = self.uncertainty_calculations.custom_uncertainty_quantification(argument="test")

        self.assertEqual(result, "custom")

    def test_custom_uncertainty_quantification_assigned_no_arguments(self):
        def custom_method():
            return "custom"

        self.uncertainty_calculations.custom_uncertainty_quantification = custom_method

        with self.assertRaises(TypeError):
            self.uncertainty_calculations.custom_uncertainty_quantification()


    def test_create_PCE_collocation_all(self):
        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_collocation()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_collocation_all_parameters(self):
        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_collocation(polynomial_order=5,
                                                                 nr_collocation_nodes=22)

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_collocation_one(self):
        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_collocation("a")

        self.assertEqual(data.uncertain_parameters, ["a"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_collocation_interpolate_error(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(1))

        model = TestingModelAdaptive()
        model.interpolate=False

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.uncertainty_calculations = UncertaintyCalculations(model=model,
                                                                parameters=parameters,
                                                                features=features,
                                                                logger_level="error")

        with self.assertRaises(ValueError):
            self.uncertainty_calculations.create_PCE_collocation()



    def test_create_PCE_collocation_rosenblatt_all(self):

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_collocation_rosenblatt()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_collocation_rosenblatt_one(self):

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_collocation_rosenblatt("a")

        self.assertEqual(data.uncertain_parameters, ["a"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_spectral_all(self):
        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_spectral()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_spectral_all_parameters(self):
        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_spectral(polynomial_order=4,
                                                              quadrature_order=6)

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_spectral_one(self):

        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_spectral("a")

        self.assertEqual(data.uncertain_parameters, ["a"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)


    def test_create_PCE_collocation_interpolate_error(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(1))

        model = TestingModelAdaptive()
        model.interpolate=False

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.uncertainty_calculations = UncertaintyCalculations(model=model,
                                                                parameters=parameters,
                                                                features=features,
                                                                logger_level="debug")

        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_spectral()
        self.assertEqual(list(U_hat.keys()), ["feature1d", "feature2d"])

    def test_create_PCE_spectral_rosenblatt_all(self):

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_spectral_rosenblatt()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)



    def test_create_PCE_spectral_rosenblatt_one(self):

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_spectral_rosenblatt("a")

        self.assertEqual(data.uncertain_parameters, ["a"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(U_hat["TestingModel1d"], cp.Poly)



    def test_calculate_sobol_first_average(self):
        data = Data(logger_level="error")

        data.add_features(["test2D", "test1D"])
        data["test2D"].sobol_first = [[4, 6], [8, 12]]
        data["test1D"].sobol_first =  [1, 2]
        data.uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.average_sensitivity(data, sensitivity="sobol_first")

        self.assertEqual(data["test2D"]["sobol_first_average"][0], 5)
        self.assertEqual(data["test2D"]["sobol_first_average"][1], 10)
        self.assertEqual(data["test1D"]["sobol_first_average"][0], 1)
        self.assertEqual(data["test1D"]["sobol_first_average"][1], 2)


    def test_calculate_first_average(self):
        data = Data(logger_level="error")

        data.add_features(["test2D", "test1D"])
        data["test2D"].sobol_first = [[4, 6], [8, 12]]
        data["test1D"].sobol_first =  [1, 2]
        data.uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.average_sensitivity(data, sensitivity="first")

        self.assertEqual(data["test2D"]["sobol_first_average"][0], 5)
        self.assertEqual(data["test2D"]["sobol_first_average"][1], 10)
        self.assertEqual(data["test1D"]["sobol_first_average"][0], 1)
        self.assertEqual(data["test1D"]["sobol_first_average"][1], 2)

    def test_calculate_total_average(self):
        data = Data(logger_level="error")

        data.add_features(["test2D", "test1D"])
        data["test2D"].sobol_total = [[4, 6], [8, 12]]
        data["test1D"].sobol_total =  [1, 2]
        data.uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.average_sensitivity(data, sensitivity="total")

        self.assertEqual(data["test2D"]["sobol_total_average"][0], 5)
        self.assertEqual(data["test2D"]["sobol_total_average"][1], 10)
        self.assertEqual(data["test1D"]["sobol_total_average"][0], 1)
        self.assertEqual(data["test1D"]["sobol_total_average"][1], 2)


    def test_calculate_sensitivity_total_average(self):
        data = Data(logger_level="error")

        data.add_features(["test2D", "test1D"])
        data["test2D"].sobol_total = [[4, 6], [8, 12]]
        data["test1D"].sobol_total = [1, 2]
        data.uncertain_parameters = ["a", "b"]

        data = self.uncertainty_calculations.average_sensitivity(data, sensitivity="sobol_total")

        self.assertEqual(data["test2D"]["sobol_total_average"][0], 5)
        self.assertEqual(data["test2D"]["sobol_total_average"][1], 10)
        self.assertEqual(data["test1D"]["sobol_total_average"][0], 1)
        self.assertEqual(data["test1D"]["sobol_total_average"][1], 2)


    def test_average_sensitivity_error(self):
        data = Data(logger_level="error")

        data.add_features(["test2D", "test1D"])
        data["test2D"].sobol_total = [[4, 6], [8, 12]]
        data["test1D"].sobol_total = [1, 2]
        data.uncertain_parameters = ["a", "b"]

        with self.assertRaises(ValueError):
            self.uncertainty_calculations.average_sensitivity(data, sensitivity="not_existing")


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
                                                                logger_level="error")

        data = Data(logger_level="error")

        q0, q1 = cp.variable(2)
        parameter_space = parameters.get_from_uncertain("distribution")
        distribution = cp.J(*parameter_space)

        data.uncertain_parameters = ["a", "b"]

        data.add_features(["TestingModel1d", "feature0d", "feature1d", "feature2d"])

        U_hat = {}
        U_hat["TestingModel1d"] = cp.Poly([q0, q1*q0, q1])
        U_hat["feature0d"] = cp.Poly([q0, q1*q0, q1])
        U_hat["feature1d"] = cp.Poly([q0, q1*q0, q1])
        U_hat["feature2d"] = cp.Poly([q0, q1*q0, q1])

        data = self.uncertainty_calculations.analyse_PCE(U_hat, distribution, data)

        # Test if all calculated properties actually exists
        data_types = ["values", "time", "mean", "variance", "percentile_5", "percentile_95",
                      "sobol_first", "sobol_first_average",
                      "sobol_total", "sobol_total_average", "labels"]

        for data_type in data_types:
            if data_type not in ["values", "time", "labels"]:
                for feature in data:
                    self.assertIsInstance(data[feature][data_type], np.ndarray)




    def test_polynomial_chaos_collocation(self):
        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])
        self.uncertainty_calculations.features = features

        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        result = subprocess.call(["h5diff", "-d", str(self.threshold), filename, compare_file])

        self.assertEqual(result, 0)



    def test_PC_collocation_rosenblatt(self):
        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])
        self.uncertainty_calculations.features = features

        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt=True,
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_Rosenblatt.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_Rosenblatt.h5")
        result = subprocess.call(["h5diff", "-d", str(self.threshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC_collocation_auto(self):
        features = TestingFeatures(features_to_run=[])
        self.uncertainty_calculations.features = features

        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt="auto",
                                                              seed=self.seed)

        self.assertNotIn("Rosenblatt", data.method)


        a = cp.Uniform(1, 2)
        b = cp.Uniform(1, 2)/a

        parameter_dict = {"a": a, "b": b}

        self.uncertainty_calculations.parameters = Parameters(parameter_dict)

        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt="auto",
                                                              seed=self.seed)

        self.assertIn("Rosenblatt", data.method)


        self.uncertainty_calculations.parameters.distribution = cp.J(cp.uniform(), cp.Uniform())

        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt="auto",
                                                              seed=self.seed)

        self.assertNotIn("Rosenblatt", data.method)


        self.uncertainty_calculations.parameters.distribution = cp.J(a, b)

        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt="auto",
                                                              seed=self.seed)

        self.assertIn("Rosenblatt", data.method)


        self.uncertainty_calculations.parameters.distribution = cp.J(a, b)
        with self.assertRaises(ValueError):
            data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                                rosenblatt=False,
                                                                seed=self.seed)


    def test_polynomial_chaos_spectral(self):
        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])
        self.uncertainty_calculations.features = features

        data = self.uncertainty_calculations.polynomial_chaos(method="spectral",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_spectral.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_spectral.h5")

        result = subprocess.call(["h5diff", "-d", str(self.threshold), filename, compare_file])

        self.assertEqual(result, 0)



    def test_polynomial_chaos_spectral_rosenblatt(self):
        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])
        self.uncertainty_calculations.features = features
        data = self.uncertainty_calculations.polynomial_chaos(method="spectral",
                                                              rosenblatt=True,
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_Rosenblatt_spectral.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_Rosenblatt_spectral.h5")

        result = subprocess.call(["h5diff", "-d", str(self.threshold), filename, compare_file])

        self.assertEqual(result, 0)



    def test_ignore_model(self):
        self.uncertainty_calculations.model.ignore = True

        data = self.uncertainty_calculations.polynomial_chaos(method="spectral",
                                                              rosenblatt=True,
                                                              seed=self.seed)
        self.assertEqual(set(data["TestingModel1d"].get_metrics()),
                         set(["evaluations", "time"]))


        data = self.uncertainty_calculations.polynomial_chaos(method="spectral",
                                                              rosenblatt=False,
                                                              seed=self.seed)
        self.assertEqual(set(data["TestingModel1d"].get_metrics()),
                             set(["evaluations", "time"]))


        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt=True,
                                                              seed=self.seed)
        self.assertEqual(set(data["TestingModel1d"].get_metrics()),
                         set(["evaluations", "time"]))


        data = self.uncertainty_calculations.polynomial_chaos(method="collocation",
                                                              rosenblatt=False,
                                                              seed=self.seed)
        self.assertEqual(set(data["TestingModel1d"].get_metrics()),
                         set(["evaluations", "time"]))


        data = self.uncertainty_calculations.monte_carlo(seed=self.seed,
                                                         nr_samples=self.nr_mc_samples)
        self.assertEqual(set(data["TestingModel1d"].get_metrics()),
                         set(["evaluations", "time"]))


    def test_ignore_model_create_PCE_spectral(self):
        np.random.seed(self.seed)

        self.uncertainty_calculations.model.ignore = True

        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_spectral()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertNotIn("TestingModel1d", U_hat)

        self.assertIn("TestingModel1d", data)


    def test_ignore_model_create_PCE_spectral_rosenblatt(self):
        np.random.seed(self.seed)

        self.uncertainty_calculations.model.ignore = True

        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_spectral_rosenblatt()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertNotIn("TestingModel1d", U_hat)

        self.assertIn("TestingModel1d", data)



    def test_ignore_model_create_PCE_collocation(self):
        np.random.seed(self.seed)

        self.uncertainty_calculations.model.ignore = True

        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_collocation()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertNotIn("TestingModel1d", U_hat)

        self.assertIn("TestingModel1d", data)


    def test_ignore_model_create_PCE_collocation_rosenblatt(self):
        np.random.seed(self.seed)

        self.uncertainty_calculations.model.ignore = True

        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_collocation_rosenblatt()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertIsInstance(U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(U_hat["feature2d"], cp.Poly)
        self.assertNotIn("TestingModel1d", U_hat)

        self.assertIn("TestingModel1d", data)

    def test_ignore_network(self):
        self.model = TestingModelAdaptive(ignore=True)
        self.model.interpolate = False

        self.uncertainty_calculations = UncertaintyCalculations(model=self.model,
                                                                parameters=self.parameters,
                                                                logger_level="error")

        U_hat, distribution, data = self.uncertainty_calculations.create_PCE_collocation()

        self.assertEqual(data.uncertain_parameters, ["a", "b"])
        self.assertNotIn("TestingModelAdaptive", U_hat)

        self.assertIn("TestingModelAdaptive", data)
        self.assertEqual(len(data["TestingModelAdaptive"].time), len(data["TestingModelAdaptive"].evaluations))


    def test_PC_custom(self):
        with self.assertRaises(NotImplementedError):
            self.uncertainty_calculations.polynomial_chaos(method="custom",
                                                           seed=self.seed)


    def test_PCE_custom_assigned(self):
        def create_PCE_custom(self, uncertain_parameters=None, custom_argument=None):
            uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

            data = Data()

            q0, q1 = cp.variable(2)
            parameter_space = [cp.Uniform(), cp.Uniform()]
            distribution = cp.J(*parameter_space)

            data.uncertain_parameters = ["a", "b"]

            data.test_value = custom_argument
            data.add_features(["TestingModel1d", "feature0d", "feature1d", "feature2d"])

            U_hat = {}
            U_hat["TestingModel1d"] = cp.Poly([q0, q1*q0, q1])
            U_hat["feature0d"] = cp.Poly([q0, q1*q0, q1])
            U_hat["feature1d"] = cp.Poly([q0, q1*q0, q1])
            U_hat["feature2d"] = cp.Poly([q0, q1*q0, q1])

            return U_hat, distribution, data


        self.uncertainty_calculations.create_PCE_custom = create_PCE_custom

        data = self.uncertainty_calculations.polynomial_chaos(method="custom",
                                                              seed=self.seed,
                                                              custom_argument="test")

        self.assertTrue(data.test_value, "test")


        # Test if all calculated properties actually exists
        data_types = ["values", "time", "mean", "variance", "percentile_5", "percentile_95",
                      "sobol_first", "sobol_first_average",
                      "sobol_total", "sobol_total_average", "labels"]

        for data_type in data_types:
            if data_type not in ["values", "time", "labels"]:
                for feature in data:
                    self.assertIsInstance(data[feature][data_type], np.ndarray)


    def test_PC_error(self):
        with self.assertRaises(ValueError):
            self.uncertainty_calculations.polynomial_chaos(method="not implemented",
                                                           seed=self.seed)


    def test_PC_parameter_a(self):
        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])
        self.uncertainty_calculations.features = features

        data = self.uncertainty_calculations.polynomial_chaos(uncertain_parameters="a",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-a.h5")
        result = subprocess.call(["h5diff", "-d", str(self.threshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC_parameter_b(self):
        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])
        self.uncertainty_calculations.features = features

        data = self.uncertainty_calculations.polynomial_chaos(uncertain_parameters="b",
                                                              seed=self.seed)

        filename = os.path.join(self.output_test_dir, "UncertaintyCalculations_single-parameter-b.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))

        compare_file = os.path.join(folder, "data/UncertaintyCalculations_single-parameter-b.h5")
        result = subprocess.call(["h5diff", "-d", str(self.threshold), filename, compare_file])

        self.assertEqual(result, 0)



    def test_ignore_model_monte_carlo(self):
        self.uncertainty_calculations.model.ignore = True

        data = self.uncertainty_calculations.monte_carlo(seed=self.seed, nr_samples=self.nr_mc_samples)
        self.assertEqual(set(data["TestingModel1d"].get_metrics()),
                         set(["evaluations", "time"]))



    def test_monte_carlo(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d_var", "feature1d_var", "feature2d_var"])

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                logger_level="error")


        data = self.uncertainty_calculations.monte_carlo(nr_samples=10, seed=10)

        # Rough tests
        self.assertTrue(np.allclose(data["TestingModel1d"]["mean"],
                                    np.arange(0, 10) + 3, atol=0.1))
        self.assertTrue(np.allclose(data["TestingModel1d"]["variance"],
                                    np.zeros(10), atol=0.2))
        self.assertTrue(np.all(np.less(data["TestingModel1d"]["percentile_5"],
                                       np.arange(0, 10) + 3)))
        self.assertTrue(np.all(np.greater(data["TestingModel1d"]["percentile_95"],
                                          np.arange(0, 10) + 3)))

        # TODO: currently no tests for the values of the sobol indices
        self.assertEqual(np.shape(data["TestingModel1d"]["sobol_first"]), (2, 10))
        self.assertEqual(np.shape(data["TestingModel1d"]["sobol_total"]), (2, 10))
        self.assertEqual(np.shape(data["TestingModel1d"]["sobol_first_average"]), (2,))
        self.assertEqual(np.shape(data["TestingModel1d"]["sobol_total_average"]), (2,))


        # Compare to pregenerated data
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        data.save(filename)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        result = subprocess.call(["h5diff", filename, compare_file])

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
                                                                logger_level="error")



        data = self.uncertainty_calculations.monte_carlo(nr_samples=self.nr_mc_samples, seed=10)

        self.assertTrue(np.array_equal(data["feature0d"]["mean"],
                                       features.feature0d(None, None)[1]))
        self.assertEqual(data["feature0d"]["variance"], 0)
        self.assertTrue(np.array_equal(data["feature0d"]["percentile_5"],
                                       features.feature0d(None, None)[1]))
        self.assertTrue(np.array_equal(data["feature0d"]["percentile_95"],
                                       features.feature0d(None, None)[1]))

        self.assertEqual(np.shape(data["feature0d"]["sobol_first"]), (2,))
        self.assertEqual(np.shape(data["feature0d"]["sobol_total"]), (2,))
        self.assertEqual(np.shape(data["feature0d"]["sobol_first_average"]), (2,))
        self.assertEqual(np.shape(data["feature0d"]["sobol_total_average"]), (2,))


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
                                                                logger_level="error")



        data = self.uncertainty_calculations.monte_carlo(nr_samples=self.nr_mc_samples, seed=10)

        self.assertTrue(np.array_equal(data["feature1d"]["mean"],
                                       features.feature1d(None, None)[1]))
        self.assertTrue(np.array_equal(data["feature1d"]["variance"],
                                       np.zeros(10)))
        self.assertTrue(np.array_equal(data["feature1d"]["percentile_5"],
                                       features.feature1d(None, None)[1]))
        self.assertTrue(np.array_equal(data["feature1d"]["percentile_95"],
                                       features.feature1d(None, None)[1]))

        self.assertEqual(np.shape(data["feature1d"]["sobol_first"]), (2, 10))
        self.assertEqual(np.shape(data["feature1d"]["sobol_total"]), (2, 10))
        self.assertEqual(np.shape(data["feature1d"]["sobol_first_average"]), (2,))
        self.assertEqual(np.shape(data["feature1d"]["sobol_total_average"]), (2,))




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
                                                                logger_level="error")


        data = self.uncertainty_calculations.monte_carlo(nr_samples=self.nr_mc_samples)

        self.assertTrue(np.array_equal(data["feature2d"]["mean"],
                                       features.feature2d(None, None)[1]))
        self.assertTrue(np.array_equal(data["feature2d"]["variance"],
                                       np.zeros((2, 10))))
        self.assertTrue(np.array_equal(data["feature2d"]["percentile_5"],
                                       features.feature2d(None, None)[1]))
        self.assertTrue(np.array_equal(data["feature2d"]["percentile_95"],
                                       features.feature2d(None, None)[1]))

        self.assertEqual(np.shape(data["feature2d"]["sobol_first"]), (2, 2, 10))
        self.assertEqual(np.shape(data["feature2d"]["sobol_total"]), (2, 2, 10))
        self.assertEqual(np.shape(data["feature2d"]["sobol_first_average"]), (2,))
        self.assertEqual(np.shape(data["feature2d"]["sobol_total_average"]), (2,))




    def test_monte_carlo_incomplete_false(self):
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
                                                                logger_level="error")

        data = self.uncertainty_calculations.monte_carlo(nr_samples=self.nr_mc_samples,
                                                         allow_incomplete=False,
                                                         seed=10)

        self.assertEqual(data.incomplete, ["TestingModelIncomplete"])
        self.assertIsNone(data["TestingModelIncomplete"].mean)
        self.assertIsNone(data["TestingModelIncomplete"].variance)
        self.assertIsNone(data["TestingModelIncomplete"].percentile_5)
        self.assertIsNone(data["TestingModelIncomplete"].percentile_95)
        self.assertIsNone(data["TestingModelIncomplete"].sobol_first)
        self.assertIsNone(data["TestingModelIncomplete"].sobol_total)
        self.assertIsNone(data["TestingModelIncomplete"].sobol_first_average)
        self.assertIsNone(data["TestingModelIncomplete"].sobol_total_average)



    def test_monte_carlo_incomplete_true(self):
        np.random.seed(self.seed)

        def model(a, b):
            if a < 1:
                return [1, 2, 3], [a, a + b, b]
            else:
                return [1, 2, 3], [a, None, b]


        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        features = TestingFeatures(features_to_run=None)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                parameters=parameters,
                                                                features=features,
                                                                logger_level="error")

        data = self.uncertainty_calculations.monte_carlo(nr_samples=self.nr_mc_samples,
                                                         allow_incomplete=True,
                                                         seed=10)

        self.assertEqual(data.incomplete, ["model"])

        # Rough bounds
        self.assertTrue(np.allclose(data["model"]["mean"],
                                    [1, 3, 2], atol=0.5))
        self.assertTrue(np.allclose(data["model"]["variance"],
                                    [0, 0, 0], atol=0.15))
        self.assertTrue(np.all(np.greater(data["model"]["percentile_5"],
                                          [0.75, 2.25, 1.5])))
        self.assertTrue(np.all(np.less(data["model"]["percentile_95"],
                                          [1.5, 4.5, 2.5])))

        self.assertEqual(np.shape(data["model"]["sobol_first"]), (2, 3))
        self.assertEqual(np.shape(data["model"]["sobol_total"]), (2, 3))
        self.assertEqual(np.shape(data["model"]["sobol_first_average"]), (2,))
        self.assertEqual(np.shape(data["model"]["sobol_total_average"]), (2,))

        self.assertFalse(np.any(np.isnan(data["model"]["sobol_first"])))
        self.assertFalse(np.any(np.isnan(data["model"]["sobol_total"])))
        self.assertFalse(np.any(np.isnan(data["model"]["sobol_first_average"])))
        self.assertFalse(np.any(np.isnan(data["model"]["sobol_total_average"])))



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
                                                                logger_level="error")

        U_hat, distribution, data =  \
            self.uncertainty_calculations.create_PCE_collocation(["a", "b"],
                                                                 allow_incomplete=False)

        self.assertEqual(U_hat, {})
        self.assertEqual(data.incomplete, ["TestingModelIncomplete"])



        # # TODO Find a good way to test this case when not all runs contains None
        # self.uncertainty_calculations = UncertaintyCalculations(model,
        #                                                         parameters=parameters,
        #                                                         features=features,
        #                                                         logger_level="error",
        #                                                         seed=self.seed,
        #                                                         allow_incomplete=True)

        # self.uncertainty_calculations.create_PCE_collocation(["a", "b"])

        # self.assertIn("TestingModelIncomplete", self.uncertainty_calculations.U_hat)
        # self.assertEqual(self.uncertainty_calculations.data.incomplete,
        #                  ["TestingModelIncomplete"])




    def test_create_PCE_spectral_rosenblatt_incomplete_false(self):
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
                                                                logger_level="error")

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_spectral(["a", "b"],
                                                              allow_incomplete=False)

        self.assertEqual(U_hat, {})
        self.assertEqual(data.incomplete, ["TestingModelIncomplete"])


    def test_create_PCE_spectral_incomplete_true(self):
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
                                                                logger_level="error")

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_spectral(["a", "b"])

        self.assertEqual(U_hat, {})
        self.assertEqual(data.incomplete, ["TestingModelIncomplete"])


    def test_create_PCE_spectral_incomplete_false(self):
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
                                                                logger_level="error")

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_spectral(["a", "b"],
                                                               allow_incomplete=False)

        self.assertEqual(U_hat, {})
        self.assertEqual(data.incomplete, ["TestingModelIncomplete"])


    def test_create_PCE_spectral_rosenblatt_incomplete_true(self):
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
                                                                logger_level="error")

        U_hat, distribution, data = \
            self.uncertainty_calculations.create_PCE_spectral(["a", "b"],
                                                              allow_incomplete=False)

        self.assertEqual(U_hat, {})
        self.assertEqual(data.incomplete, ["TestingModelIncomplete"])


    def separate_output_values_use_case(self, base_evaluation):
         # N = 1, D = 1 => Nt = 3
        nr_uncertain_parameters = 1
        nr_samples = 1

        evaluations = [base_evaluation + 1, base_evaluation + 2, base_evaluation + 3]

        A, B, AB = self.uncertainty_calculations.separate_output_values(evaluations,
                                                                        nr_uncertain_parameters=nr_uncertain_parameters,
                                                                        nr_samples=nr_samples)

        original_A = [base_evaluation + 1]
        original_B = [base_evaluation + 3]
        original_AB = [[base_evaluation + 2]]

        self.assertTrue(np.array_equal(A, original_A))
        self.assertTrue(np.array_equal(B, original_B))
        self.assertTrue(np.array_equal(AB, original_AB))

        # N = 1, D = 2 => Nt = 4
        nr_uncertain_parameters = 2
        nr_samples = 1

        evaluations = [base_evaluation + 1, base_evaluation + 2, base_evaluation + 3, base_evaluation + 4]

        A, B, AB = self.uncertainty_calculations.separate_output_values(evaluations,
                                                                        nr_uncertain_parameters=nr_uncertain_parameters,
                                                                        nr_samples=nr_samples)
        original_A = [base_evaluation + 1,]
        original_B = [base_evaluation + 4]
        original_AB = [[base_evaluation + 2,  base_evaluation + 3]]


        self.assertTrue(np.array_equal(A, original_A))
        self.assertTrue(np.array_equal(B, original_B))
        self.assertTrue(np.array_equal(AB, original_AB))


        # N = 2, D = 2 => Nt = 8
        nr_uncertain_parameters = 2
        nr_samples = 2
        evaluations = [base_evaluation + 1, base_evaluation + 2, base_evaluation + 3, base_evaluation + 4,
                       base_evaluation + 5, base_evaluation + 6, base_evaluation + 7, base_evaluation + 8]

        A, B, AB = self.uncertainty_calculations.separate_output_values(evaluations,
                                                                        nr_uncertain_parameters=nr_uncertain_parameters,
                                                                        nr_samples=nr_samples)
        original_A = [base_evaluation + 1, base_evaluation + 5]
        original_B = [base_evaluation + 4, base_evaluation + 8]
        original_AB = [[base_evaluation + 2, base_evaluation + 3],
                       [base_evaluation + 6, base_evaluation + 7]]


        self.assertTrue(np.array_equal(A, original_A))
        self.assertTrue(np.array_equal(B, original_B))
        self.assertTrue(np.array_equal(AB, original_AB))


    def test_separate_output_values_0D(self):
        # N = 1, D = 1 => Nt = 3
        nr_uncertain_parameters = 1
        nr_samples = 1

        evaluations = [1, 2, 3]

        A, B, AB = self.uncertainty_calculations.separate_output_values(evaluations,
                                                                        nr_uncertain_parameters=nr_uncertain_parameters,
                                                                        nr_samples=nr_samples)

        original_A, original_B, original_AB, _ = separate_output_values(np.array(evaluations),
                                                                     D=nr_uncertain_parameters,
                                                                     N=nr_samples,
                                                                     calc_second_order=False)

        self.assertTrue(np.array_equal(A, original_A))
        self.assertTrue(np.array_equal(B, original_B))
        self.assertTrue(np.array_equal(AB, original_AB))

        # N = 1, D = 2 => Nt = 4
        nr_uncertain_parameters = 2
        nr_samples = 1

        evaluations = [1, 2, 3, 4]

        A, B, AB = self.uncertainty_calculations.separate_output_values(evaluations,
                                                                        nr_uncertain_parameters=nr_uncertain_parameters,
                                                                        nr_samples=nr_samples)

        original_A, original_B, original_AB, _ = separate_output_values(np.array(evaluations),
                                                                     D=nr_uncertain_parameters,
                                                                     N=nr_samples,
                                                                     calc_second_order=False)

        self.assertTrue(np.array_equal(A, original_A))
        self.assertTrue(np.array_equal(B, original_B))
        self.assertTrue(np.array_equal(AB, original_AB))


        # N = 2, D = 2 => Nt = 8
        nr_uncertain_parameters = 2
        nr_samples = 2
        evaluations = [1, 2, 3, 4,
                       5, 6, 7, 8]

        A, B, AB = self.uncertainty_calculations.separate_output_values(evaluations,
                                                                        nr_uncertain_parameters=nr_uncertain_parameters,
                                                                        nr_samples=nr_samples)

        original_A, original_B, original_AB, _ = separate_output_values(np.array(evaluations),
                                                                     D=nr_uncertain_parameters,
                                                                     N=nr_samples,
                                                                     calc_second_order=False)

        self.assertTrue(np.array_equal(A, original_A))
        self.assertTrue(np.array_equal(B, original_B))
        self.assertTrue(np.array_equal(AB, original_AB))


    def test_separate_output_values(self):

        test_arrays = [0, np.zeros((4)), np.zeros((4, 3)), np.zeros((4, 3, 4, 5, 6, 2, 3, 1, 2))]

        for test_array in test_arrays:
            self.separate_output_values_use_case(test_array)



    def mc_calculate_sobol_use_case(self, base_evaluation):
        # N = 1, D = 1 => Nt = 3
        nr_uncertain_parameters = 1
        nr_samples = 1

        evaluations = [base_evaluation + 1, base_evaluation + 2, base_evaluation + 3]

        sobol_first, sobol_total = self.uncertainty_calculations.mc_calculate_sobol(evaluations=evaluations,
                                                                                    nr_uncertain_parameters=nr_uncertain_parameters,
                                                                                    nr_samples=nr_samples)

        self.assertEqual(np.shape(sobol_first[0]), np.shape(base_evaluation))
        self.assertEqual(np.shape(sobol_total[0]), np.shape(base_evaluation))


        # N = 1, D = 2 => Nt = 4
        nr_uncertain_parameters = 2
        nr_samples = 1

        evaluations = [base_evaluation + 1, base_evaluation + 2, base_evaluation + 3, base_evaluation + 4]

        sobol_first, sobol_total = self.uncertainty_calculations.mc_calculate_sobol(evaluations=evaluations,
                                                                                    nr_uncertain_parameters=nr_uncertain_parameters,
                                                                                    nr_samples=nr_samples)


        self.assertEqual(np.shape(sobol_first[0]), np.shape(base_evaluation))
        self.assertEqual(np.shape(sobol_first[1]), np.shape(base_evaluation))

        self.assertEqual(np.shape(sobol_total[0]), np.shape(base_evaluation))
        self.assertEqual(np.shape(sobol_total[1]), np.shape(base_evaluation))


        # N = 2, D = 2 => Nt = 8
        nr_uncertain_parameters = 2
        nr_samples = 2

        evaluations = [base_evaluation + 1, base_evaluation + 2, base_evaluation + 3, base_evaluation + 4,
                       base_evaluation + 5, base_evaluation + 6, base_evaluation + 7, base_evaluation + 8]

        sobol_first, sobol_total = self.uncertainty_calculations.mc_calculate_sobol(evaluations=evaluations,
                                                                                    nr_uncertain_parameters=nr_uncertain_parameters,
                                                                                    nr_samples=nr_samples)

        self.assertEqual(np.shape(sobol_first[0]), np.shape(base_evaluation))
        self.assertEqual(np.shape(sobol_first[1]), np.shape(base_evaluation))

        self.assertEqual(np.shape(sobol_total[0]), np.shape(base_evaluation))
        self.assertEqual(np.shape(sobol_total[1]), np.shape(base_evaluation))



    def test_mc_calculate_sobol(self):
        test_arrays = [0, np.zeros((4)), np.zeros((4, 3)), np.zeros((4, 3, 4, 5, 6, 2, 3, 1, 2))]

        for test_array in test_arrays:
            self.mc_calculate_sobol_use_case(test_array)