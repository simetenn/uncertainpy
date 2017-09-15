import unittest
import os
import shutil
import subprocess
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')

from uncertainpy import UncertaintyEstimation
from uncertainpy.parameters import Parameters
from uncertainpy.features import GeneralFeatures
from uncertainpy import uniform, normal
from uncertainpy import UncertaintyCalculations
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

        self.parameterlist = [["a", 1, None],
                              ["b", 2, None]]

        self.parameters = Parameters(self.parameterlist)
        self.parameters.set_all_distributions(uniform(0.5))

        self.model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyEstimation(self.model,
                                                 parameters=self.parameters,
                                                 features=features,
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed,
                                                 nr_mc_samples=10)

        self.figureformat = ".png"


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        uncertainty = UncertaintyEstimation(self.model, self.parameters)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.parameters, Parameters)


    def test_init_parameterlist(self):
        uncertainty = UncertaintyEstimation(self.model, self.parameterlist)

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.parameters, Parameters)

    def test_init_parameter_error(self):

        with self.assertRaises(TypeError):
            UncertaintyEstimation(self.model, 2)


    def test_intit_features(self):
        uncertainty = UncertaintyEstimation(self.model,
                                            self.parameters,
                                            verbose_level="error")
        self.assertIsInstance(uncertainty.features, GeneralFeatures)

        uncertainty = UncertaintyEstimation(self.model,
                                            self.parameters,
                                            features=TestingFeatures(),
                                            verbose_level="error")
        self.assertIsInstance(uncertainty.features, TestingFeatures)




    def test_init_uncertainty_calculations(self):

        class TempUncertaintyCalculations(UncertaintyCalculations):
            def create_PCE_custom(self):
                "custom PCE method"

        uncertainty = UncertaintyEstimation(
            self.model,
            self.parameters,
            uncertainty_calculations=TempUncertaintyCalculations(self.model),
            verbose_level="error",
            output_dir_data=self.output_test_dir,
            output_dir_figures=self.output_test_dir
        )

        self.assertIsInstance(uncertainty.uncertainty_calculations, TempUncertaintyCalculations)


    def test_set_parameters(self):
        uncertainty = UncertaintyEstimation(model=TestingModel1d(),
                                            parameters=None,
                                            verbose_level="error",
                                            seed=self.seed)
        uncertainty.parameters = Parameters()

        self.assertIsInstance(uncertainty.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_list(self):
        uncertainty = UncertaintyEstimation(model=TestingModel1d(),
                                            parameters=None,
                                            verbose_level="error",
                                            seed=self.seed)
        uncertainty.parameters = self.parameterlist

        self.assertIsInstance(uncertainty.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.parameters, Parameters)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.parameters,
                              Parameters)


    def test_set_parameter_error(self):
        uncertainty = UncertaintyEstimation(model=TestingModel1d(),
                                            parameters=None,
                                            verbose_level="error",
                                            seed=self.seed)
        with self.assertRaises(TypeError):
            uncertainty.parameters = 2


    def test_set_features(self):
        uncertainty = UncertaintyEstimation(model=TestingModel1d(),
                                            parameters=None,
                                            verbose_level="error",
                                            seed=self.seed)
        uncertainty.features = TestingFeatures()

        self.assertIsInstance(uncertainty.features, TestingFeatures)
        self.assertIsInstance(uncertainty.uncertainty_calculations.features, TestingFeatures)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.features,
                              TestingFeatures)


    def test_feature_function(self):
        def feature_function(t, U):
            return "t", "U"

        self.uncertainty.features = feature_function
        self.assertIsInstance(self.uncertainty.features, GeneralFeatures)

        t, U = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        self.assertEqual(self.uncertainty.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(t, U):
            return "t", "U"

        def feature_function2(t, U):
            return "t2", "U2"


        self.uncertainty.features = [feature_function, feature_function2]
        self.assertIsInstance(self.uncertainty.features, GeneralFeatures)

        t, U = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")


        t, U = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.uncertainty.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

        self.assertEqual(self.uncertainty.features.features_to_run,
                         ["feature_function", "feature_function2"])



    def test_feature_functions_base(self):
        def feature_function(t, U):
            return "t", "U"

        def feature_function2(t, U):
            return "t2", "U2"

        implemented_features = ["nr_spikes", "time_before_first_spike",
                                "spike_rate", "average_AP_overshoot",
                                "average_AHP_depth", "average_AP_width",
                                "accommodation_index"]

        self.uncertainty.features = SpikingFeatures([feature_function, feature_function2])
        self.assertIsInstance(self.uncertainty.features, SpikingFeatures)

        t, U = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.uncertainty.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

        self.assertEqual(set(self.uncertainty.features.features_to_run),
                         set(["feature_function", "feature_function2"] + implemented_features))


    def test_set_model(self):
        uncertainty = UncertaintyEstimation(model=TestingModel1d(),
                                            parameters=None,
                                            verbose_level="error",
                                            seed=self.seed)
        uncertainty.model = TestingModel1d()

        self.assertIsInstance(uncertainty.model, TestingModel1d)
        self.assertIsInstance(uncertainty.uncertainty_calculations.model, TestingModel1d)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.model,
                              TestingModel1d)


    def test_set_model_function(self):
        uncertainty = UncertaintyEstimation(model=model_function,
                                            parameters=None,
                                            verbose_level="error",
                                            seed=self.seed)

        self.assertIsInstance(uncertainty.model, Model)
        self.assertIsInstance(uncertainty.uncertainty_calculations.model, Model)
        self.assertIsInstance(uncertainty.uncertainty_calculations.runmodel.model,
                              Model)



    # def test_label(self):
    #     uncertainty = UncertaintyEstimation(model=TestingModel1d(),
    #                                         parameters=None,
    #                                         verbose_level="error",
    #                                         seed=self.seed)



    def test_create_PCE_custom(self):

        def create_PCE_custom(self, uncertain_parameters=None):
            self.data = Data()
            self.test_value = "custom PCE method"

        uncertainty = UncertaintyEstimation(
            self.model,
            self.parameters,
            create_PCE_custom=create_PCE_custom,
            verbose_level="error",
            output_dir_data=self.output_test_dir,
            output_dir_figures=self.output_test_dir
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
        self.uncertainty.polynomial_chaos_single()

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
        self.uncertainty.polynomial_chaos()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/model_function.h5")
        filename = os.path.join(self.output_test_dir, "model_function.h5")
        self.assertTrue(os.path.isfile(filename))


        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)


    def test_polynomial_chaos(self):
        self.uncertainty.polynomial_chaos()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC_plot(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 save_data=False,
                                                 save_figures=True,
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed,
                                                 nr_mc_samples=10)


        self.uncertainty.polynomial_chaos()

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_polynomial_chaos_single_plot(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 save_data=False,
                                                 save_figures=True,
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed,
                                                 nr_mc_samples=10)

        self.uncertainty.polynomial_chaos_single()

        self.compare_plot("TestingModel1d_single-parameter-a/TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-a/TestingModel1d_confidence-interval")

        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_confidence-interval")

        self.compare_plot("TestingModel1d_single-parameter-b/TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-b/TestingModel1d_confidence-interval")

        self.compare_plot("TestingModel1d_single-parameter-b/feature1d_mean-variance")
        self.compare_plot("TestingModel1d_single-parameter-b/feature1d_confidence-interval")


    def test_monte_carlo_single(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed,
                                                 nr_mc_samples=10)


        self.uncertainty.monte_carlo_single(filename="TestingModel1d_MC")


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

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 parameters=parameters,
                                                 features=features,
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed,
                                                 nr_mc_samples=10**1)


        self.uncertainty.monte_carlo(filename="TestingModel1d_MC")


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)


    def test_load(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        self.uncertainty.load(os.path.join(folder, "data", "test_save_mock"))

        for data_type in self.uncertainty.data.data_types:
            if data_type == "labels":
                continue
            else:
                self.assertTrue(np.array_equal(self.uncertainty.data["feature1d"][data_type], [1., 2.]))
                self.assertTrue(np.array_equal(self.uncertainty.data["TestingModel1d"][data_type], [3., 4.]))

        self.assertEqual(self.uncertainty.data.uncertain_parameters, ["a", "b"])

        self.assertTrue(np.array_equal(self.uncertainty.data["TestingModel1d"]["labels"], ["xlabel", "ylabel"]))
        self.assertTrue(np.array_equal(self.uncertainty.data["feature1d"]["labels"], ["xlabel", "ylabel"]))



    def test_plot_all(self):
        self.uncertainty.polynomial_chaos()
        self.uncertainty.plot(condensed=False, sensitivity="sensitivity_1")


        self.compare_plot("TestingModel1d_mean")
        self.compare_plot("TestingModel1d_variance")
        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")

        self.compare_plot("TestingModel1d_sensitivity_1_a")
        self.compare_plot("TestingModel1d_sensitivity_1_b")
        self.compare_plot("TestingModel1d_sensitivity_1")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")
        self.compare_plot("feature1d_sensitivity_1")
        self.compare_plot("feature1d_sensitivity_1_grid")
        self.compare_plot("feature0d_total-sensitivity_1")

        self.compare_plot("TestingModel1d_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")

        self.compare_plot("feature1d_sensitivity_t_a")
        self.compare_plot("feature1d_sensitivity_t_b")
        self.compare_plot("feature1d_sensitivity_t")
        self.compare_plot("feature1d_sensitivity_t_grid")



        self.compare_plot("TestingModel1d_sensitivity_t_a")
        self.compare_plot("TestingModel1d_sensitivity_t_b")
        self.compare_plot("TestingModel1d_sensitivity_t")
        self.compare_plot("TestingModel1d_sensitivity_t_grid")

        self.compare_plot("feature0d_total-sensitivity_t")


        self.compare_plot("TestingModel1d_total-sensitivity_t")
        self.compare_plot("feature0d_total-sensitivity_t")
        self.compare_plot("feature1d_total-sensitivity_t")
        self.compare_plot("feature2d_total-sensitivity_t")



        self.compare_plot("total-sensitivity_t_grid")
        self.compare_plot("total-sensitivity_1_grid")


    def test_plot_condensed(self):
        self.uncertainty.polynomial_chaos()
        self.uncertainty.plot()

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")



    def test_plotNoSensitivity(self):
        self.uncertainty.polynomial_chaos()
        self.uncertainty.plot(condensed=False, sensitivity=None)

        self.compare_plot("TestingModel1d_mean")
        self.compare_plot("TestingModel1d_variance")
        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")


    def test_results(self):
        self.uncertainty.polynomial_chaos()
        self.uncertainty.plot(results=True)


        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature0d_results/*.png")))
        self.assertEqual(plot_count, 1)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature1d_results/*.png")))
        self.assertEqual(plot_count, self.uncertainty.uncertainty_calculations.nr_pc_samples)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature2d_results/*.png")))
        self.assertEqual(plot_count, self.uncertainty.uncertainty_calculations.nr_pc_samples)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/TestingModel1d_results/*.png")))
        self.assertEqual(plot_count, self.uncertainty.uncertainty_calculations.nr_pc_samples)



    def test_PCresults(self):
        self.uncertainty.polynomial_chaos(plot_results=True)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature0d_results/*.png")))
        self.assertEqual(plot_count, 1)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature1d_results/*.png")))
        self.assertEqual(plot_count, self.uncertainty.uncertainty_calculations.nr_pc_samples)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature2d_results/*.png")))
        self.assertEqual(plot_count, self.uncertainty.uncertainty_calculations.nr_pc_samples)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/TestingModel1d_results/*.png")))
        self.assertEqual(plot_count, self.uncertainty.uncertainty_calculations.nr_pc_samples)


    def set_up_test_calculations(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.set_all_distributions(uniform(0.5))

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=None)

        self.uncertainty = UncertaintyEstimation(model,
                                                 parameters=parameters,
                                                 features=features,
                                                 uncertainty_calculations=TestingUncertaintyCalculations(model),
                                                 save_data=False,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error")



    def test_uncertainty_quantificationPCAll(self):
        self.set_up_test_calculations()

        self.uncertainty.uncertainty_quantification(method="pc", plot_condensed=False)

        self.assertEqual(self.uncertainty.data["function"], "PC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], ["a", "b"])
        self.assertEqual(self.uncertainty.data["method"], "regression")
        self.assertEqual(self.uncertainty.data["rosenblatt"], False)


    def test_uncertainty_quantification_PC_single_rosenblatt(self):
        self.set_up_test_calculations()

        self.uncertainty.uncertainty_quantification(method="pc",
                                                    pc_method="regression",
                                                    plot_condensed=True,
                                                    single=True,
                                                    rosenblatt=True)

        self.assertEqual(self.uncertainty.data["function"], "PC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], "b")
        self.assertEqual(self.uncertainty.data["method"], "regression")
        self.assertEqual(self.uncertainty.data["rosenblatt"], True)


    def test_uncertainty_quantification_monte_carlo(self):
        self.set_up_test_calculations()

        self.uncertainty.uncertainty_quantification(method="mc", plot_condensed=False)

        self.assertEqual(self.uncertainty.data["function"], "MC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], ["a", "b"])


    def test_uncertainty_quantification_monte_carlo_single(self):
        self.set_up_test_calculations()

        self.uncertainty.uncertainty_quantification(method="mc", plot_condensed=False, single=True)

        self.assertEqual(self.uncertainty.data["function"], "MC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], "b")



    def test_uncertainty_quantification_custom(self):
        self.set_up_test_calculations()

        self.uncertainty.uncertainty_quantification(method="custom", custom_keyword="value")

        self.assertEqual(self.uncertainty.data["function"], "custom_uncertainty_quantification")
        self.assertEqual(self.uncertainty.data["custom_keyword"], "value")


    def test_custom_uncertainty_quantification(self):
        self.set_up_test_calculations()

        self.uncertainty.custom_uncertainty_quantification(custom_keyword="value")

        self.assertEqual(self.uncertainty.data["function"], "custom_uncertainty_quantification")
        self.assertEqual(self.uncertainty.data["custom_keyword"], "value")



if __name__ == "__main__":
    unittest.main()
