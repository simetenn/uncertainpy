import unittest
import os
import shutil
import subprocess
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')

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

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyQuantification(self.model,
                                                     parameters=self.parameters,
                                                     features=features,
                                                     verbose_level="error")

        self.figureformat = ".png"
        self.nr_mc_samples = 10**1



    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        uncertainty = UncertaintyQuantification(self.model, self.parameters)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.parameters, Parameters)


    def test_init_parameter_list(self):
        uncertainty = UncertaintyQuantification(self.model, self.parameter_list)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.parameters, Parameters)

    def test_init_parameter_error(self):

        with self.assertRaises(TypeError):
            UncertaintyQuantification(self.model, 2)


    def test_init_features(self):
        uncertainty = UncertaintyQuantification(self.model,
                                                self.parameters,
                                                verbose_level="error")
        self.assertIsInstance(uncertainty.features, Features)

        uncertainty = UncertaintyQuantification(self.model,
                                                self.parameters,
                                                features=TestingFeatures(),
                                                verbose_level="error")
        self.assertIsInstance(uncertainty.features, TestingFeatures)




    def test_init_uncertainty_calculations(self):

        class TempUncertaintyCalculations(UncertaintyCalculations):
            def create_PCE_custom(self):
                "custom PCE method"

        uncertainty = UncertaintyQuantification(
            self.model,
            self.parameters,
            uncertainty_calculations=TempUncertaintyCalculations(self.model),
            verbose_level="error"
        )

        self.assertIsInstance(uncertainty.uncertainty_calculations, TempUncertaintyCalculations)


    def test_set_parameters(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                verbose_level="error")
        uncertainty.parameters = Parameters()

        self.assertIsInstance(uncertainty.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_list(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                verbose_level="error")
        uncertainty.parameters = self.parameter_list

        self.assertIsInstance(uncertainty.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_error(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                verbose_level="error")
        with self.assertRaises(TypeError):
            uncertainty.parameters = 2


    def test_set_features(self):
        uncertainty = UncertaintyQuantification(model=TestingModel1d(),
                                                parameters=None,
                                                verbose_level="error")
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
                                                verbose_level="error")
        uncertainty.model = TestingModel1d()

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.model,
                              TestingModel1d)


    def test_set_model_function(self):
        uncertainty = UncertaintyQuantification(model=model_function,
                                                parameters=None,
                                                verbose_level="error")

        self.assertIsInstance(uncertainty.model, Model)
        self.assertIsInstance(uncertainty.uncertainty_calculations.model, Model)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.model,
                              Model)



    # def test_label(self):
    #     uncertainty = UncertaintyQuantification(model=TestingModel1d(),
    #                                         parameters=None,
    #                                         verbose_level="error",
    #                                         seed=self.seed)



    def test_create_PCE_custom(self):

        def create_PCE_custom(self, uncertain_parameters=None):
            self.data = Data()
            self.test_value = "custom PCE method"

        uncertainty = UncertaintyQuantification(
            self.model,
            self.parameters,
            create_PCE_custom=create_PCE_custom,
            verbose_level="error"
        )


        uncertainty.polynomial_chaos(method="custom")
        self.assertTrue(uncertainty.uncertainty_calculations.test_value,
                        "custom PCE method")

    # def test_convert_uncertain_parameters_list(self):
    #     result = self.uncertainty.convert_uncertain_parameters(["a", "b"])

    #     self.assertEqual(result, ["a", "b"])

    # def test_convert_uncertain_parameters_string(self):
    #     result = self.uncertainty.convert_uncertain_parameters("a")

    #     self.assertEqual(result, ["a"])


    # def test_convert_uncertain_parameters_none(self):
    #     result = self.uncertainty.convert_uncertain_parameters(None)

    #     self.assertEqual(result, ["a", "b"])


    def test_polynomial_chaos_single(self):
        self.uncertainty.polynomial_chaos_single(data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

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
        self.uncertainty.polynomial_chaos(data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/model_function.h5")
        filename = os.path.join(self.output_test_dir, "model_function.h5")
        self.assertTrue(os.path.isfile(filename))


        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)


    def test_polynomial_chaos(self):
        self.uncertainty.polynomial_chaos(data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

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

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyQuantification(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 verbose_level="error")


        self.uncertainty.polynomial_chaos(plot="condensed_sensitivity_1",
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_prediction-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("sensitivity_1_sum_grid")


    def test_polynomial_chaos_single_plot(self):
        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyQuantification(model,
                                                     features=features,
                                                     parameters=parameters,
                                                     verbose_level="error")

        self.uncertainty.polynomial_chaos_single(plot="condensed_sensitivity_1",
                                                 data_folder=self.output_test_dir,
                                                 figure_folder=self.output_test_dir,
                                                 seed=self.seed)

        self.compare_plot("TestingModel1d_single-parameter-a/TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-a/TestingModel1d_prediction-interval")

        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_prediction-interval")

        self.compare_plot("TestingModel1d_single-parameter-b/TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-b/TestingModel1d_prediction-interval")

        self.compare_plot("TestingModel1d_single-parameter-b/feature1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-b/feature1d_prediction-interval")


    def test_monte_carlo_single(self):
        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        parameters = Parameters(parameter_list)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyQuantification(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 verbose_level="error")


        self.uncertainty.monte_carlo_single(filename="TestingModel1d_MC",
                                            plot=None,
                                            data_folder=self.output_test_dir,
                                            seed=self.seed,
                                            nr_samples=self.nr_mc_samples)


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

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyQuantification(model,
                                                     parameters=parameters,
                                                     features=features,
                                                     verbose_level="error")


        self.uncertainty.monte_carlo(filename="TestingModel1d_MC",
                                     plot=None,
                                     data_folder=self.output_test_dir,
                                     seed=self.seed,
                                     nr_samples=self.nr_mc_samples)


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_load(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        self.uncertainty.load(os.path.join(folder, "data", "test_save_mock"))

        data_types = ["values", "time", "mean", "variance", "percentile_5", "percentile_95",
                      "sensitivity_1", "sensitivity_1_sum",
                      "sensitivity_t", "sensitivity_t_sum"]

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

        self.uncertainty.plot(type="all")


        self.compare_plot("TestingModel1d_mean")
        self.compare_plot("TestingModel1d_variance")
        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")

        self.compare_plot("TestingModel1d_sensitivity_1_a")
        self.compare_plot("TestingModel1d_sensitivity_1_b")
        self.compare_plot("TestingModel1d_sensitivity_1")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_prediction-interval")

        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")
        self.compare_plot("feature1d_sensitivity_1")
        self.compare_plot("feature1d_sensitivity_1_grid")
        self.compare_plot("feature0d_sensitivity_1_sum")

        self.compare_plot("TestingModel1d_sensitivity_1_sum")
        self.compare_plot("feature0d_sensitivity_1_sum")
        self.compare_plot("feature1d_sensitivity_1_sum")
        self.compare_plot("feature2d_sensitivity_1_sum")

        self.compare_plot("feature1d_sensitivity_t_a")
        self.compare_plot("feature1d_sensitivity_t_b")
        self.compare_plot("feature1d_sensitivity_t")
        self.compare_plot("feature1d_sensitivity_t_grid")



        self.compare_plot("TestingModel1d_sensitivity_t_a")
        self.compare_plot("TestingModel1d_sensitivity_t_b")
        self.compare_plot("TestingModel1d_sensitivity_t")
        self.compare_plot("TestingModel1d_sensitivity_t_grid")

        self.compare_plot("feature0d_sensitivity_t_sum")


        self.compare_plot("TestingModel1d_sensitivity_t_sum")
        self.compare_plot("feature0d_sensitivity_t_sum")
        self.compare_plot("feature1d_sensitivity_t_sum")
        self.compare_plot("feature2d_sensitivity_t_sum")



        self.compare_plot("sensitivity_t_sum_grid")
        self.compare_plot("sensitivity_1_sum_grid")


    def test_plot_condensed(self):
        self.uncertainty.polynomial_chaos(plot=None,
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)
        self.uncertainty.plot(type="condensed_sensitivity_1")

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_prediction-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("sensitivity_1_sum_grid")



    def test_plotNoSensitivity(self):
        self.uncertainty.polynomial_chaos(plot=None,
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        self.uncertainty.plot(type="condensed_no_sensitivity")

        self.compare_plot("TestingModel1d_mean")
        self.compare_plot("TestingModel1d_variance")
        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_prediction-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_prediction-interval")


    def test_plot_evaluations(self):
        self.uncertainty.polynomial_chaos(plot=None,
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        self.uncertainty.plot(type="evaluations", folder=self.output_test_dir)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature0d_evaluations/*.png")))
        self.assertEqual(plot_count, 1)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature1d_evaluations/*.png")))
        self.assertEqual(plot_count, 22)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature2d_evaluations/*.png")))
        self.assertEqual(plot_count, 22)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/TestingModel1d_evaluations/*.png")))
        self.assertEqual(plot_count, 22)



    def test_PCevaluations(self):
        self.uncertainty.polynomial_chaos(nr_collocation_nodes=12,
                                          plot="evaluations",
                                          data_folder=self.output_test_dir,
                                          figure_folder=self.output_test_dir,
                                          seed=self.seed)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature0d_evaluations/*.png")))
        self.assertEqual(plot_count, 1)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature1d_evaluations/*.png")))
        self.assertEqual(plot_count, 12)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "evaluations/feature2d_evaluations/*.png")))
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
                                                     verbose_level="error")



    def test_quantifyPCAll(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="pc",
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



    def test_quantify_PC_single_rosenblatt(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="pc",
                                  pc_method="collocation",
                                  single=True,
                                  rosenblatt=True,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed)

        self.assertEqual(self.uncertainty.data.arguments["function"], "PC")
        self.assertEqual(self.uncertainty.data.arguments["uncertain_parameters"], "b")
        self.assertEqual(self.uncertainty.data.arguments["method"], "collocation")
        self.assertEqual(self.uncertainty.data.arguments["rosenblatt"], True)


    def test_quantify_monte_carlo(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="mc",
                                  nr_mc_samples=self.nr_mc_samples,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed)

        self.assertEqual(self.uncertainty.data.arguments["function"], "MC")
        self.assertEqual(self.uncertainty.data.arguments["uncertain_parameters"], ["a", "b"])
        self.assertEqual(self.uncertainty.data.arguments["seed"], self.seed)
        self.assertEqual(self.uncertainty.data.arguments["nr_samples"], self.nr_mc_samples)


    def test_quantify_monte_carlo_single(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="mc",
                                  single=True,
                                  nr_mc_samples=self.nr_mc_samples,
                                  data_folder=self.output_test_dir,
                                  figure_folder=self.output_test_dir,
                                  seed=self.seed)

        self.assertEqual(self.uncertainty.data.arguments["function"], "MC")
        self.assertEqual(self.uncertainty.data.arguments["uncertain_parameters"], "b")



    def test_quantify_custom(self):
        self.set_up_test_calculations()

        self.uncertainty.quantify(method="custom", custom_keyword="value")

        self.assertEqual(self.uncertainty.data.arguments["function"], "custom_uncertainty_quantification")
        self.assertEqual(self.uncertainty.data.arguments["custom_keyword"], "value")


    def test_custom_uncertainty_quantification(self):
        self.set_up_test_calculations()

        self.uncertainty.custom_uncertainty_quantification(custom_keyword="value")

        self.assertEqual(self.uncertainty.data.arguments["function"], "custom_uncertainty_quantification")
        self.assertEqual(self.uncertainty.data.arguments["custom_keyword"], "value")



if __name__ == "__main__":
    unittest.main()
