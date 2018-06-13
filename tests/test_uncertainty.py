import unittest
import os
import shutil
import subprocess
import glob
import numpy as np
import chaospy as cp
import logging

# import matplotlib
# matplotlib.use('Agg')

from uncertainpy import UncertaintyQuantification
from uncertainpy.parameters import Parameters
from uncertainpy.features import Features
from uncertainpy import uniform, normal
from uncertainpy.core import UncertaintyCalculations
from uncertainpy import Data
from uncertainpy import Model
from uncertainpy import SpikingFeatures


from .testing_classes import TestingFeatures
from .testing_classes import TestingModel1d, model_function
from .testing_classes import TestingUncertaintyCalculations
from .testing_classes import TestCasePlot


class TestUncertainty(TestCasePlot):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10
        self.difference_treshold = 1e-10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.parameter_list = [["a", 1, None],
                               ["b", 2, None]]

        self.parameters = Parameters(self.parameter_list)
        self.parameters.set_all_distributions(uniform(0.5))

        self.model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])

        self.uncertainty = UncertaintyQuantification(self.model,
                                                     parameters=self.parameters,
                                                     features=features,
                                                     logger_level="error",
                                                     logger_filename=None)

        self.figureformat = ".png"
        self.nr_mc_samples = 10**1



    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        uncertainty = UncertaintyQuantification(self.model,
                                                self.parameters,
                                                logger_level="error",
                                                logger_filename=None)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.parameters, Parameters)


    def test_init_parameter_list(self):
        uncertainty = UncertaintyQuantification(self.model,
                                                self.parameter_list,
                                                logger_level="error",
                                                logger_filename=None)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.parameters, Parameters)

    def test_init_parameter_error(self):

        with self.assertRaises(TypeError):
            UncertaintyQuantification(self.model,
                                      2,
                                      logger_level="error",
                                      logger_filename=None)


    def test_init_backend_error(self):
        with self.assertRaises(ValueError):
            UncertaintyQuantification(self.model, self.parameter_list, backend="not a backend")


    def test_init_features(self):
        uncertainty = UncertaintyQuantification(self.model,
                                                self.parameters,
                                                logger_level="error",
                                                logger_filename=None)
        self.assertIsInstance(uncertainty.features, Features)

        uncertainty = UncertaintyQuantification(self.model,
                                                self.parameters,
                                                features=TestingFeatures(),
                                                logger_level="error",
                                                logger_filename=None)
        self.assertIsInstance(uncertainty.features, TestingFeatures)



    def test_init_uncertainty_calculations_custom(self):
        def uq(self):
            return "uq"

        def pce(self, uncertain_parameters=None):
            return "pce"

        uncertainty = UncertaintyQuantification(
            self.model,
            self.parameters,
            create_PCE_custom=pce,
            custom_uncertainty_quantification=uq,
            logger_level="error"
        )

        result = uncertainty.uncertainty_calculations.create_PCE_custom()
        self.assertEqual(result, "pce")

        result = uncertainty.uncertainty_calculations.custom_uncertainty_quantification()
        self.assertEqual(result, "uq")

    def test_init_uncertainty_calculations(self):

        class TempUncertaintyCalculations(UncertaintyCalculations):
            def create_PCE_custom(self):
                "custom PCE method"

        uncertainty = UncertaintyQuantification(
            self.model,
            self.parameters,
            uncertainty_calculations=TempUncertaintyCalculations(self.model),
            logger_level="error",
            logger_filename=None
        )

        self.assertIsInstance(uncertainty.uncertainty_calculations, TempUncertaintyCalculations)


    def test_set_parameters(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                logger_level="error",
                                                logger_filename=None)
        uncertainty.parameters = Parameters()

        self.assertIsInstance(uncertainty.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_list(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                logger_level="error",
                                                logger_filename=None)
        uncertainty.parameters = self.parameter_list

        self.assertIsInstance(uncertainty.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_error(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                logger_level="error",
                                                logger_filename=None)
        with self.assertRaises(TypeError):
            uncertainty.parameters = 2


    def test_set_features(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                logger_level="error",
                                                logger_filename=None)
        uncertainty.features = TestingFeatures()

        self.assertIsInstance(uncertainty.features, TestingFeatures)
        self.assertIsInstance(uncertainty.uncertainty_calculations.features, TestingFeatures)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.features,
                              TestingFeatures)


    def test_feature_function(self):
        def feature_function(time, values):
            return "time", "values"

        self.uncertainty.features = feature_function
        self.assertIsInstance(self.uncertainty.features, Features)

        time, values = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        self.assertEqual(self.uncertainty.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(time, values):
            return "time", "values"

        def feature_function2(time, values):
            return "t2", "U2"


        self.uncertainty.features = [feature_function, feature_function2]
        self.assertIsInstance(self.uncertainty.features, Features)

        time, values = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")


        time, values = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        time, values = self.uncertainty.features.feature_function2(None, None)
        self.assertEqual(time, "t2")
        self.assertEqual(values, "U2")

        self.assertEqual(self.uncertainty.features.features_to_run,
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

        self.uncertainty.features = SpikingFeatures([feature_function, feature_function2])
        self.assertIsInstance(self.uncertainty.features, SpikingFeatures)

        time, values = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        time, values = self.uncertainty.features.feature_function2(None, None)
        self.assertEqual(time, "time2")
        self.assertEqual(values, "values2")

        self.assertEqual(set(self.uncertainty.features.features_to_run),
                         set(["feature_function", "feature_function2"] + implemented_features))


    def test_set_model(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                logger_level="error",
                                                logger_filename=None)
        uncertainty.model = TestingModel1d()

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.model,
                              TestingModel1d)


    def test_set_model_function(self):
        uncertainty = UncertaintyQuantification(model=model_function,
                                                parameters=None,
                                                logger_level="error",
                                                logger_filename=None)

        self.assertIsInstance(uncertainty.model, Model)
        self.assertIsInstance(uncertainty.uncertainty_calculations.model, Model)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.model,
                              Model)






    def test_create_PCE_custom(self):

        def create_PCE_custom(self, uncertain_parameters=None):
            data = Data()

            q0, q1 = cp.variable(2)
            parameter_space = [cp.Uniform(), cp.Uniform()]
            distribution = cp.J(*parameter_space)

            data.uncertain_parameters = ["a", "b"]

            data.test_value = "custom PCE method"
            data.add_features(["TestingModel1d", "feature0d_var", "feature1d_var", "feature2d_var"])

            U_hat = {}
            U_hat["TestingModel1d"] = cp.Poly([q0, q1*q0, q1])
            U_hat["feature0d_var"] = cp.Poly([q0, q1*q0, q1])
            U_hat["feature1d_var"] = cp.Poly([q0, q1*q0, q1])
            U_hat["feature2d_var"] = cp.Poly([q0, q1*q0, q1])

            return U_hat, distribution, data

        uncertainty = UncertaintyQuantification(
            model=self.model,
            parameters=self.parameters,
            create_PCE_custom=create_PCE_custom,
            logger_level="error",
            logger_filename=None
        )

        data = uncertainty.polynomial_chaos(method="custom", data_folder=self.output_test_dir, figure_folder=self.output_test_dir)

        self.assertTrue(uncertainty.data.test_value,
                        "custom PCE method")

        self.assertTrue(data.test_value,
                        "custom PCE method")

        # Test if all calculated properties actually exists
        data_types = ["evaluations", "time", "mean", "variance", "percentile_5", "percentile_95",
                      "sobol_first", "sobol_first_average",
                      "sobol_total", "sobol_total_average", "labels"]

        for data_type in data_types:
            if data_type not in ["evaluations", "time", "labels"]:
                for feature in uncertainty.data:
                    self.assertIsInstance(uncertainty.data[feature][data_type], np.ndarray)

        for data_type in data_types:
            if data_type not in ["evaluations", "time", "labels"]:
                for feature in data:
                    self.assertIsInstance(data[feature][data_type], np.ndarray)

    def test_convert_uncertain_parameters_list(self):
        result = self.uncertainty.uncertainty_calculations.convert_uncertain_parameters(["a", "b"])

        self.assertEqual(result, ["a", "b"])

    def test_convert_uncertain_parameters_string(self):
        result = self.uncertainty.uncertainty_calculations.convert_uncertain_parameters("a")

        self.assertEqual(result, ["a"])


    def test_convert_uncertain_parameters_none(self):
        result = self.uncertainty.uncertainty_calculations.convert_uncertain_parameters(None)

        self.assertEqual(result, ["a", "b"])


    def test_polynomial_chaos_single(self):
        data_dict = self.uncertainty.polynomial_chaos_single(data_folder=self.output_test_dir,
                                                             figure_folder=self.output_test_dir,
                                                             seed=self.seed)

        self.assertIsInstance(data_dict["a"], Data)
        self.assertIsInstance(data_dict["b"], Data)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-a.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])


        self.assertEqual(result, 0)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-b.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-b.h5")
        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC_model_function(self):
        self.uncertainty.model = model_function
        data = self.uncertainty.polynomial_chaos(data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

        self.assertIsInstance(data, Data)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/model_function.h5")
        filename = os.path.join(self.output_test_dir, "model_function.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)


    def test_polynomial_chaos(self):
        data = self.uncertainty.polynomial_chaos(data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

        self.assertIsInstance(data, Data)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC_plot(self):
        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])

        self.uncertainty = UncertaintyQuantification(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 logger_level="error",
                                                 logger_filename=None)


        self.uncertainty.polynomial_chaos(plot="condensed_first",
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")
        self.compare_plot("TestingModel1d_sobol_first_grid")

        self.compare_plot("feature1d_var_mean-variance")
        self.compare_plot("feature1d_var_prediction-interval")
        self.compare_plot("feature1d_var_sobol_first_grid")

        self.compare_plot("feature0d_var_sobol_first")

        self.compare_plot("sobol_first_average_grid")


    def test_polynomial_chaos_single_plot(self):
        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])

        self.uncertainty = UncertaintyQuantification(model,
                                                     features=features,
                                                     parameters=parameters,
                                                     logger_level="error",
                                                     logger_filename=None)

        self.uncertainty.polynomial_chaos_single(plot="condensed_first",
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

        self.compare_plot("TestingModel1d_single-parameter-a/TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-a/TestingModel1d_prediction-interval")

        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_var_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_var_prediction-interval")

        self.compare_plot("TestingModel1d_single-parameter-b/TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-b/TestingModel1d_prediction-interval")

        self.compare_plot("TestingModel1d_single-parameter-b/feature1d_var_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-b/feature1d_var_prediction-interval")


    def test_var_monte_carlo_single(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])

        self.uncertainty = UncertaintyQuantification(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 logger_level="error",
                                                 logger_filename=None)


        data_dict = self.uncertainty.monte_carlo_single(filename="TestingModel1d_MC",
                                                        plot=None,
                                                        data_folder=self.output_test_dir,
                                                        seed=self.seed,
                                                        nr_samples=self.nr_mc_samples)

        self.assertIsInstance(data_dict["a"], Data)
        self.assertIsInstance(data_dict["b"], Data)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC_single-parameter-a.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC_single-parameter-a.h5")

        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


        compare_file = os.path.join(folder, "data/TestingModel1d_MC_single-parameter-b.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC_single-parameter-b.h5")

        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_monte_carlo(self):

        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])

        self.uncertainty = UncertaintyQuantification(model,
                                                     parameters=parameters,
                                                     features=features,
                                                     logger_level="error",
                                                     logger_filename=None)


        data = self.uncertainty.monte_carlo(filename="TestingModel1d_MC",
                                            plot=None,
                                            data_folder=self.output_test_dir,
                                            seed=self.seed,
                                            nr_samples=self.nr_mc_samples)

        self.assertIsInstance(data, Data)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_load(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        self.uncertainty.load(os.path.join(folder, "data", "test_save_mock"))

        data_types = ["evaluations", "time", "mean", "variance", "percentile_5", "percentile_95",
                      "sobol_first", "sobol_first_average",
                      "sobol_total", "sobol_total_average"]

        for data_type in data_types:
            self.assertTrue(np.array_equal(self.uncertainty.data["feature1d"][data_type], [1., 2.]))
            self.assertTrue(np.array_equal(self.uncertainty.data["TestingModel1d"][data_type], [3., 4.]))

        self.assertEqual(self.uncertainty.data.uncertain_parameters, ["a", "b"])

        self.assertTrue(np.array_equal(self.uncertainty.data["TestingModel1d"]["labels"], ["xlabel", "ylabel"]))
        self.assertTrue(np.array_equal(self.uncertainty.data["feature1d"]["labels"], ["xlabel", "ylabel"]))



    def test_plot_all(self):
        self.uncertainty.polynomial_chaos(plot=None,
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        self.uncertainty.plot(type="all", folder=self.output_test_dir)


        self.compare_plot("TestingModel1d_mean")
        self.compare_plot("TestingModel1d_variance")
        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")

        self.compare_plot("TestingModel1d_sobol_first_a")
        self.compare_plot("TestingModel1d_sobol_first_b")
        self.compare_plot("TestingModel1d_sobol_first")
        self.compare_plot("TestingModel1d_sobol_first_grid")


        self.compare_plot("feature1d_var_mean")
        self.compare_plot("feature1d_var_variance")
        self.compare_plot("feature1d_var_mean-variance")
        self.compare_plot("feature1d_var_prediction-interval")

        self.compare_plot("feature1d_var_sobol_first_a")
        self.compare_plot("feature1d_var_sobol_first_b")
        self.compare_plot("feature1d_var_sobol_first")
        self.compare_plot("feature1d_var_sobol_first_grid")
        self.compare_plot("feature0d_var_sobol_first_average")

        self.compare_plot("TestingModel1d_sobol_first_average")
        self.compare_plot("feature0d_var_sobol_first_average")
        self.compare_plot("feature1d_var_sobol_first_average")
        self.compare_plot("feature2d_var_sobol_first_average")

        self.compare_plot("feature1d_var_sobol_total_a")
        self.compare_plot("feature1d_var_sobol_total_b")
        self.compare_plot("feature1d_var_sobol_total")
        self.compare_plot("feature1d_var_sobol_total_grid")



        self.compare_plot("TestingModel1d_sobol_total_a")
        self.compare_plot("TestingModel1d_sobol_total_b")
        self.compare_plot("TestingModel1d_sobol_total")
        self.compare_plot("TestingModel1d_sobol_total_grid")

        self.compare_plot("feature0d_var_sobol_total_average")


        self.compare_plot("TestingModel1d_sobol_total_average")
        self.compare_plot("feature0d_var_sobol_total_average")
        self.compare_plot("feature1d_var_sobol_total_average")
        self.compare_plot("feature2d_var_sobol_total_average")


        self.compare_plot("sobol_total_average_grid")
        self.compare_plot("sobol_first_average_grid")


    def test_plot_condensed(self):
        self.uncertainty.polynomial_chaos(plot=None,
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)
        self.uncertainty.plot(type="condensed_first", folder=self.output_test_dir)

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")
        self.compare_plot("TestingModel1d_sobol_first_grid")

        self.compare_plot("feature1d_var_mean-variance")
        self.compare_plot("feature1d_var_prediction-interval")
        self.compare_plot("feature1d_var_sobol_first_grid")

        self.compare_plot("feature0d_var_sobol_first")

        self.compare_plot("sobol_first_average_grid")

        self.compare_plot("feature2d_var_mean")
        self.compare_plot("feature2d_var_variance")




    def test_plotNoSensitivity(self):
        self.uncertainty.polynomial_chaos(plot=None,
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        self.uncertainty.plot(type="condensed_no_sensitivity", folder=self.output_test_dir)

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")

        self.compare_plot("feature1d_var_mean-variance")
        self.compare_plot("feature1d_var_prediction-interval")

        self.compare_plot("feature0d_var")

        self.compare_plot("feature2d_var_mean")
        self.compare_plot("feature2d_var_variance")

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "*.png")))
        self.assertEqual(plot_count, 7)



    def test_plot_evaluations(self):
        self.uncertainty.polynomial_chaos(plot=None,
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        self.uncertainty.plot(type="evaluations", folder=self.output_test_dir)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature0d_var_evaluations/*.png")))
        self.assertEqual(plot_count, 1)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature1d_var_evaluations/*.png")))
        self.assertEqual(plot_count, 32)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature2d_var_evaluations/*.png")))
        self.assertEqual(plot_count, 32)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/TestingModel1d_evaluations/*.png")))
        self.assertEqual(plot_count, 32)



    def test_PCevaluations(self):
        self.uncertainty.polynomial_chaos(nr_collocation_nodes=12,
                                          plot="evaluations",
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature0d_var_evaluations/*.png")))
        self.assertEqual(plot_count, 1)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature1d_var_evaluations/*.png")))
        self.assertEqual(plot_count, 12)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature2d_var_evaluations/*.png")))
        self.assertEqual(plot_count, 12)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/TestingModel1d_evaluations/*.png")))
        self.assertEqual(plot_count, 12)


    def set_up_test_calculations(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty = UncertaintyQuantification(model,
                                                     parameters=parameters,
                                                     features=features,
                                                     uncertainty_calculations=TestingUncertaintyCalculations(model),
                                                     logger_level="error",
                                                     logger_filename=None)


    def test_set_uncertainty_calculations(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty = UncertaintyQuantification(model,
                                                     parameters=parameters,
                                                     features=features,
                                                     logger_level="error",
                                                     logger_filename=None)

        self.uncertainty.uncertainty_calculations = TestingUncertaintyCalculations()

        self.assertIsInstance(self.uncertainty.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(self.uncertainty.uncertainty_calculations.features, TestingFeatures)


    def test_quantifyPCAll(self):
        self.set_up_test_calculations()

        data = self.uncertainty.quantify(method="pc",
                                         data_folder=self.output_test_dir,
                                         figure_folder=self.output_test_dir,
                                         seed=self.seed,
                                         uncertain_parameters=None,
                                         pc_method="collocation",
                                         rosenblatt=False,
                                         polynomial_order=2,
                                         nr_collocation_nodes=50,
                                         quadrature_order=3,
                                         nr_pc_mc_samples=10**3,
                                         allow_incomplete=False)


        self.assertEqual(self.uncertainty.data.arguments["function"], "PC")
        self.assertEqual(self.uncertainty.data.arguments["uncertain_parameters"], ["a", "b"])
        self.assertEqual(self.uncertainty.data.arguments["method"], "collocation")
        self.assertEqual(self.uncertainty.data.arguments["rosenblatt"], False)
        self.assertEqual(self.uncertainty.data.arguments["polynomial_order"], 2)
        self.assertEqual(self.uncertainty.data.arguments["nr_collocation_nodes"], 50)
        self.assertEqual(self.uncertainty.data.arguments["quadrature_order"], 3)
        self.assertEqual(self.uncertainty.data.arguments["nr_pc_mc_samples"],10**3)
        self.assertEqual(self.uncertainty.data.arguments["allow_incomplete"], False)
        self.assertEqual(self.uncertainty.data.arguments["seed"], self.seed)

        self.assertEqual(data.arguments["function"], "PC")
        self.assertEqual(data.arguments["uncertain_parameters"], ["a", "b"])
        self.assertEqual(data.arguments["method"], "collocation")
        self.assertEqual(data.arguments["rosenblatt"], False)
        self.assertEqual(data.arguments["polynomial_order"], 2)
        self.assertEqual(data.arguments["nr_collocation_nodes"], 50)
        self.assertEqual(data.arguments["quadrature_order"], 3)
        self.assertEqual(data.arguments["nr_pc_mc_samples"],10**3)
        self.assertEqual(data.arguments["allow_incomplete"], False)
        self.assertEqual(data.arguments["seed"], self.seed)



    def test_no_save(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=False,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed)

        file_count = len(glob.glob(os.path.join(self.output_test_dir, "*")))
        self.assertEqual(file_count, 0)

        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=False,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

        file_count = len(glob.glob(os.path.join(self.output_test_dir, "*")))
        self.assertEqual(file_count, 0)


        self.uncertainty.quantify(method="mc",
                                  plot=None,
                                  save=False,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed)

        file_count = len(glob.glob(os.path.join(self.output_test_dir, "*")))
        self.assertEqual(file_count, 0)

        self.uncertainty.monte_carlo_single(plot=None,
                                            save=False,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed)

        file_count = len(glob.glob(os.path.join(self.output_test_dir, "*")))
        self.assertEqual(file_count, 0)



    def test_save_auto(self):
        self.set_up_test_calculations()
        self.uncertainty.backend = "auto"

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed)

        file_path = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

        file_path = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed)

        file_path = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))





    def test_save_auto_h5py(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed,
                                  filename="test.h5")

        file_path = os.path.join(self.output_test_dir, "test.h5")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed,
                                                 filename="test.h5")

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                            plot=None,
                            save=True,
                            data_folder=self.output_test_dir,
                            figure_folder=self.output_test_dir,
                            seed=self.seed,
                            filename="test.h5")

        file_path = os.path.join(self.output_test_dir, "test.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            filename="test.h5",
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))


    def test_save_auto_exdir(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed,
                                  filename="test.exdir")

        file_path = os.path.join(self.output_test_dir, "test.exdir")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed,
                                                 filename="test.exdir")

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-a.exdir")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-b.exdir")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                            plot=None,
                            save=True,
                            data_folder=self.output_test_dir,
                            figure_folder=self.output_test_dir,
                            seed=self.seed,
                            filename="test.exdir")

        file_path = os.path.join(self.output_test_dir, "test.exdir")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            filename="test.exdir",
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-a.exdir")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-b.exdir")
        self.assertTrue(os.path.exists(file_path))



    def test_save_auto_h5py_default(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed,
                                  filename="test")

        file_path = os.path.join(self.output_test_dir, "test.h5")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed,
                                                 filename="test")

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                            plot=None,
                            save=True,
                            data_folder=self.output_test_dir,
                            figure_folder=self.output_test_dir,
                            seed=self.seed,
                            filename="test")

        file_path = os.path.join(self.output_test_dir, "test.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            filename="test",
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))


    def test_save_backend_h5py(self):
        self.set_up_test_calculations()
        self.uncertainty.backend = "hdf5"

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed,
                                  filename="test.test")

        file_path = os.path.join(self.output_test_dir, "test.test.h5")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed,
                                                 filename="test.test")

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                            plot=None,
                            save=True,
                            data_folder=self.output_test_dir,
                            figure_folder=self.output_test_dir,
                            seed=self.seed,
                            filename="test.test")

        file_path = os.path.join(self.output_test_dir, "test.test.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            filename="test.test",
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))



    def test_save_backend_exdir(self):
        self.set_up_test_calculations()
        self.uncertainty.backend = "exdir"

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed,
                                  filename="test.test")

        file_path = os.path.join(self.output_test_dir, "test.test.exdir")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed,
                                                 filename="test.test")

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-a.exdir")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-b.exdir")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                            plot=None,
                            save=True,
                            data_folder=self.output_test_dir,
                            figure_folder=self.output_test_dir,
                            seed=self.seed,
                            filename="test.test")

        file_path = os.path.join(self.output_test_dir, "test.test.exdir")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            filename="test.test",
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-a.exdir")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.test_single-parameter-b.exdir")
        self.assertTrue(os.path.exists(file_path))


    def test_save_backend_h5py_exdir(self):
        self.set_up_test_calculations()
        self.uncertainty.backend = "hdf5"

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed,
                                  filename="test.exdir")

        file_path = os.path.join(self.output_test_dir, "test.exdir.h5")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed,
                                                 filename="test.exdir")

        file_path = os.path.join(self.output_test_dir, "test.exdir_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.exdir_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                            plot=None,
                            save=True,
                            data_folder=self.output_test_dir,
                            figure_folder=self.output_test_dir,
                            seed=self.seed,
                            filename="test.exdir")

        file_path = os.path.join(self.output_test_dir, "test.exdir.h5")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            filename="test.exdir",
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "test.exdir_single-parameter-a.h5")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.exdir_single-parameter-b.h5")
        self.assertTrue(os.path.exists(file_path))



    def test_save_backend_exdir_h5py(self):
        self.set_up_test_calculations()
        self.uncertainty.backend = "exdir"

        self.uncertainty.quantify(method="pc",
                                  plot=None,
                                  save=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed,
                                  filename="test.h5")

        file_path = os.path.join(self.output_test_dir, "test.h5.exdir")
        self.assertTrue(os.path.exists(file_path))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.polynomial_chaos_single(plot=None,
                                                 save=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed,
                                                 filename="test.h5")

        file_path = os.path.join(self.output_test_dir, "test.h5_single-parameter-a.exdir")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.h5_single-parameter-b.exdir")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.quantify(method="mc",
                            plot=None,
                            save=True,
                            data_folder=self.output_test_dir,
                            figure_folder=self.output_test_dir,
                            seed=self.seed,
                            filename="test.h5")

        file_path = os.path.join(self.output_test_dir, "test.h5.exdir")
        self.assertTrue(os.path.exists(file_path))

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


        self.uncertainty.monte_carlo_single(plot=None,
                                            save=True,
                                            data_folder=self.output_test_dir,
                                            figure_folder=self.output_test_dir,
                                            seed=self.seed,
                                            filename="test.h5",
                                            nr_samples=self.nr_mc_samples)

        file_path = os.path.join(self.output_test_dir, "test.h5_single-parameter-a.exdir")
        self.assertTrue(os.path.exists(file_path))

        file_path = os.path.join(self.output_test_dir, "test.h5_single-parameter-b.exdir")
        self.assertTrue(os.path.exists(file_path))




    def test_PC_single_rosenblatt(self):
        self.set_up_test_calculations()

        self.uncertainty.polynomial_chaos_single(rosenblatt=True,
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

        self.assertEqual(self.uncertainty.data.arguments["function"], "PC")
        self.assertEqual(self.uncertainty.data.arguments["uncertain_parameters"], "b")
        self.assertEqual(self.uncertainty.data.arguments["method"], "collocation")
        self.assertEqual(self.uncertainty.data.arguments["rosenblatt"], True)


    def test_quantify_monte_carlo(self):
        self.set_up_test_calculations()

        data = self.uncertainty.quantify(method="mc",
                                         nr_mc_samples=self.nr_mc_samples,
                                         data_folder=self.output_test_dir,
                                         figure_folder=self.output_test_dir,
                                         seed=self.seed)

        self.assertEqual(self.uncertainty.data.arguments["function"], "MC")
        self.assertEqual(self.uncertainty.data.arguments["uncertain_parameters"], ["a", "b"])
        self.assertEqual(self.uncertainty.data.arguments["seed"], self.seed)
        self.assertEqual(self.uncertainty.data.arguments["nr_samples"], self.nr_mc_samples)

        self.assertEqual(data.arguments["function"], "MC")
        self.assertEqual(data.arguments["uncertain_parameters"], ["a", "b"])
        self.assertEqual(data.arguments["seed"], self.seed)
        self.assertEqual(data.arguments["nr_samples"], self.nr_mc_samples)


    def test_quantify_custom(self):
        self.set_up_test_calculations()

        data = self.uncertainty.quantify(method="custom", custom_keyword="value",
                                         data_folder=self.output_test_dir,
                                         plot=None)

        self.assertEqual(self.uncertainty.data.arguments["function"], "custom_uncertainty_quantification")
        self.assertEqual(self.uncertainty.data.arguments["custom_keyword"], "value")

        self.assertEqual(data.arguments["function"], "custom_uncertainty_quantification")
        self.assertEqual(data.arguments["custom_keyword"], "value")


    def test_custom_uncertainty_quantification(self):
        self.set_up_test_calculations()

        data = self.uncertainty.custom_uncertainty_quantification(custom_keyword="value", data_folder=self.output_test_dir, plot=None)

        self.assertEqual(self.uncertainty.data.arguments["function"], "custom_uncertainty_quantification")
        self.assertEqual(self.uncertainty.data.arguments["custom_keyword"], "value")

        self.assertEqual(data.arguments["function"], "custom_uncertainty_quantification")
        self.assertEqual(data.arguments["custom_keyword"], "value")


    def test_custom_uncertainty_quantification_arguments(self):
        def custom(self, argument):
            self.convert_uncertain_parameters()

            data = Data()
            data.argument = argument

            return data

        self.uncertainty.uncertainty_calculations.custom_uncertainty_quantification = custom

        self.uncertainty.quantify(method="custom", argument="value", plot=None, data_folder=self.output_test_dir)

        self.assertEqual(self.uncertainty.data.argument, "value")


    def test_custom_uncertainty_quantification_arguments(self):
        def custom(self, argument):
            self.convert_uncertain_parameters()

            data = Data()
            data.argument = argument

            return data

        self.uncertainty.uncertainty_calculations.custom_uncertainty_quantification = custom

        self.uncertainty.quantify(method="custom", argument="value", data_folder=self.output_test_dir, plot=None)

        self.assertEqual(self.uncertainty.data.argument, "value")




    def test_logging(self):
        # remove possible added handlers
        logger = logging.getLogger("uncertainpy")

        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

        logfile = os.path.join(self.output_test_dir, "test.log")

        features = TestingFeatures(features_to_run=["feature0d_var",
                                                    "feature1d_var",
                                                    "feature2d_var"])


        self.uncertainty = UncertaintyQuantification(self.model,
                                                     parameters=self.parameters,
                                                     features=features,
                                                     logger_level="debug",
                                                     logger_filename=logfile)

        self.uncertainty.quantify()

        self.assertEqual(len(open(logfile).readlines()), 1)

        # remove the handler we have added
        logger = logging.getLogger("uncertainpy")

        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
