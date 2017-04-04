import unittest
import os
import shutil
import subprocess
import numpy as np
import glob

from uncertainpy import UncertaintyEstimation
from uncertainpy.parameters import Parameters
from uncertainpy.features import GeneralFeatures
from uncertainpy import Distribution
from uncertainpy import UncertaintyCalculations
from uncertainpy import Data
from uncertainpy import Model
from uncertainpy import NeuronFeatures


from testing_classes import TestingFeatures
from testing_classes import TestingModel1d, model_function
from testing_classes import TestingUncertaintyCalculations


class TestUncertainty(unittest.TestCase):
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
        self.parameters.setAllDistributions(Distribution(0.5).uniform)

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
            uncertainty = UncertaintyEstimation(self.model, 2)


    def test_intitFeatures(self):
        uncertainty = UncertaintyEstimation(self.model,
                                            self.parameters,
                                            verbose_level="error")
        self.assertIsInstance(uncertainty.features, GeneralFeatures)

        uncertainty = UncertaintyEstimation(self.model,
                                            self.parameters,
                                            features=TestingFeatures(),
                                            verbose_level="error")
        self.assertIsInstance(uncertainty.features, TestingFeatures)




    def test_initUncertaintyCalculations(self):

        class TempUncertaintyCalculations(UncertaintyCalculations):
            def PCECustom(self):
                "custom PCE method"

        uncertainty = UncertaintyEstimation(
            self.model,
            self.parameters,
            uncertainty_calculations=TempUncertaintyCalculations(self.model),
            verbose_level="error"
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

        implemented_features = ["nrSpikes", "timeBeforeFirstSpike",
                                "spikeRate", "averageAPOvershoot",
                                "averageAHPDepth", "averageAPWidth",
                                "accomondationIndex"]

        self.uncertainty.base_features = NeuronFeatures
        self.uncertainty.features = [feature_function, feature_function2]
        self.assertIsInstance(self.uncertainty.features, NeuronFeatures)

        t, U = self.uncertainty.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

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



    def test_label(self):
        uncertainty = UncertaintyEstimation(model=TestingModel1d(),
                                            parameters=None,
                                            verbose_level="error",
                                            xlabel="xlabel",
                                            ylabel="ylabel",
                                            seed=self.seed)

        self.assertEqual(uncertainty.model.xlabel, "xlabel")
        self.assertEqual(uncertainty.model.ylabel, "ylabel")

    def test_PCECustom(self):

        def PCECustom(self, uncertain_parameters=None):
            self.data = Data()
            self.test_value = "custom PCE method"

        uncertainty = UncertaintyEstimation(
            self.model,
            self.parameters,
            PCECustom=PCECustom,
            verbose_level="error"
        )


        uncertainty.PC(method="custom")
        self.assertTrue(uncertainty.uncertainty_calculations.test_value,
                        "custom PCE method")

    def test_convertUncertainParametersList(self):
        result = self.uncertainty.convertUncertainParameters(["a", "b"])

        self.assertEqual(result, ["a", "b"])

    def test_convertUncertainParametersString(self):
        result = self.uncertainty.convertUncertainParameters("a")

        self.assertEqual(result, ["a"])


    def test_convertUncertainParametersNone(self):
        result = self.uncertainty.convertUncertainParameters(None)

        self.assertEqual(result, ["a", "b"])




    def test_PCSingle(self):


        self.uncertainty.PCSingle()

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
        self.uncertainty.PC()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/model_function.h5")
        filename = os.path.join(self.output_test_dir, "model_function.h5")
        self.assertTrue(os.path.isfile(filename))


        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold),
                                  filename, compare_file])

        self.assertEqual(result, 0)


    def test_PC(self):
        self.uncertainty.PC()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)



    def test_PCPlot(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])


        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 save_data=False,
                                                 save_figures=True,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed,
                                                 nr_mc_samples=10)


        self.uncertainty.PC()

        self.plot_exists("directComparison_mean-variance")
        self.plot_exists("directComparison_confidence-interval")
        self.plot_exists("directComparison_sensitivity_1_grid")

        self.plot_exists("feature1d_mean-variance")
        self.plot_exists("feature1d_confidence-interval")
        self.plot_exists("feature1d_sensitivity_1_grid")

        self.plot_exists("feature0d_sensitivity_1")


        self.plot_exists("total-sensitivity_1_grid")



    def test_PCSinglePlot(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

        model = TestingModel1d()

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 parameters=parameters,
                                                 save_data=False,
                                                 save_figures=True,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed,
                                                 nr_mc_samples=10)



        self.uncertainty.PCSingle()

        self.plot_exists("TestingModel1d_single-parameter-a/directComparison_mean-variance")
        self.plot_exists("TestingModel1d_single-parameter-a/directComparison_confidence-interval")

        self.plot_exists("TestingModel1d_single-parameter-a/feature1d_mean-variance")
        self.plot_exists("TestingModel1d_single-parameter-a/feature1d_confidence-interval")


        self.plot_exists("TestingModel1d_single-parameter-b/directComparison_mean-variance")
        self.plot_exists("TestingModel1d_single-parameter-b/directComparison_confidence-interval")

        self.plot_exists("TestingModel1d_single-parameter-b/feature1d_mean-variance")
        self.plot_exists("TestingModel1d_single-parameter-b/feature1d_confidence-interval")




    def test_MCSingle(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

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


        self.uncertainty.MCSingle(filename="TestingModel1d_MC")


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



    def test_MC(self):

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

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


        self.uncertainty.MC(filename="TestingModel1d_MC")


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", "-d", str(self.difference_treshold), filename, compare_file])

        self.assertEqual(result, 0)




    def test_load(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        self.uncertainty.load(os.path.join(folder, "data", "test_save_mock"))

        self.assertTrue(np.array_equal(self.uncertainty.data.U["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.U["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.E["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.E["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.t["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.t["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.Var["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.Var["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_05["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_05["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_95["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_95["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.sensitivity_1["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.sensitivity_1["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.total_sensitivity_1["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.total_sensitivity_1["directComparison"],
                                       [3., 4.]))

        self.assertTrue(np.array_equal(self.uncertainty.data.sensitivity_t["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.sensitivity_t["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.total_sensitivity_t["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.uncertainty.data.total_sensitivity_t["directComparison"],
                                       [3., 4.]))


        self.assertEqual(self.uncertainty.data.uncertain_parameters[0], "a")
        self.assertEqual(self.uncertainty.data.uncertain_parameters[1], "b")

        self.assertEqual(self.uncertainty.data.xlabel, "xlabel")
        self.assertEqual(self.uncertainty.data.ylabel, "ylabel")

        self.assertEqual(self.uncertainty.data.feature_list[0], "directComparison")
        self.assertEqual(self.uncertainty.data.feature_list[1], "feature1")



    def test_plotAll(self):
        self.uncertainty.PC()
        self.uncertainty.plot(condensed=False, sensitivity=True)


        self.plot_exists("directComparison_mean")
        self.plot_exists("directComparison_variance")
        self.plot_exists("directComparison_mean-variance")
        self.plot_exists("directComparison_confidence-interval")

        self.plot_exists("directComparison_sensitivity_1_a")
        self.plot_exists("directComparison_sensitivity_1_b")
        self.plot_exists("directComparison_sensitivity_1")
        self.plot_exists("directComparison_sensitivity_1_grid")


        self.plot_exists("feature1d_mean")
        self.plot_exists("feature1d_variance")
        self.plot_exists("feature1d_mean-variance")
        self.plot_exists("feature1d_confidence-interval")

        self.plot_exists("feature1d_sensitivity_1_a")
        self.plot_exists("feature1d_sensitivity_1_b")
        self.plot_exists("feature1d_sensitivity_1")
        self.plot_exists("feature1d_sensitivity_1_grid")
        self.plot_exists("feature0d_total-sensitivity_1")

        self.plot_exists("directComparison_total-sensitivity_1")
        self.plot_exists("feature0d_total-sensitivity_1")
        self.plot_exists("feature1d_total-sensitivity_1")
        self.plot_exists("feature2d_total-sensitivity_1")

        self.plot_exists("feature1d_sensitivity_t_a")
        self.plot_exists("feature1d_sensitivity_t_b")
        self.plot_exists("feature1d_sensitivity_t")
        self.plot_exists("feature1d_sensitivity_t_grid")



        self.plot_exists("directComparison_sensitivity_t_a")
        self.plot_exists("directComparison_sensitivity_t_b")
        self.plot_exists("directComparison_sensitivity_t")
        self.plot_exists("directComparison_sensitivity_t_grid")

        self.plot_exists("feature0d_total-sensitivity_t")


        self.plot_exists("directComparison_total-sensitivity_t")
        self.plot_exists("feature0d_total-sensitivity_t")
        self.plot_exists("feature1d_total-sensitivity_t")
        self.plot_exists("feature2d_total-sensitivity_t")



        self.plot_exists("total-sensitivity_t_grid")
        self.plot_exists("total-sensitivity_1_grid")


    def test_plotCondensed(self):
        self.uncertainty.PC()
        self.uncertainty.plot()

        self.plot_exists("directComparison_mean-variance")
        self.plot_exists("directComparison_confidence-interval")
        self.plot_exists("directComparison_sensitivity_1_grid")

        self.plot_exists("feature1d_mean-variance")
        self.plot_exists("feature1d_confidence-interval")
        self.plot_exists("feature1d_sensitivity_1_grid")

        self.plot_exists("feature0d_sensitivity_1")

        self.plot_exists("total-sensitivity_1_grid")



    def test_plotNoSensitivity(self):
        self.uncertainty.PC()
        self.uncertainty.plot(condensed=False, sensitivity=False)

        self.plot_exists("directComparison_mean")
        self.plot_exists("directComparison_variance")
        self.plot_exists("directComparison_mean-variance")
        self.plot_exists("directComparison_confidence-interval")


        self.plot_exists("feature1d_mean")
        self.plot_exists("feature1d_variance")
        self.plot_exists("feature1d_mean-variance")
        self.plot_exists("feature1d_confidence-interval")



    def test_plotSimulatorResults(self):
        self.uncertainty.PC()
        self.uncertainty.plot(simulator_results=True)


        self.assertEqual(len(glob.glob(os.path.join(self.output_test_dir,
                                                    "simulator_results/*.png"))),
                         self.uncertainty.uncertainty_calculations.nr_pc_samples)



    def test_PCplotSimulatorResults(self):
        self.uncertainty.PC(plot_simulator_results=True)

        self.assertEqual(len(glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png"))),
                         self.uncertainty.uncertainty_calculations.nr_pc_samples)



    def setUpTestCalculations(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

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



    def test_UQPCAll(self):
        self.setUpTestCalculations()

        self.uncertainty.UQ(method="pc", plot_condensed=False)

        self.assertEqual(self.uncertainty.data["function"], "PC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], ["a", "b"])
        self.assertEqual(self.uncertainty.data["method"], "regression")
        self.assertEqual(self.uncertainty.data["rosenblatt"], False)


    def test_UQPCSingleResultRosenblatt(self):
        self.setUpTestCalculations()

        self.uncertainty.UQ(method="pc",
                            pc_method="regression",
                            plot_condensed=True,
                            single=True,
                            rosenblatt=True)

        self.assertEqual(self.uncertainty.data["function"], "PC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], "b")
        self.assertEqual(self.uncertainty.data["method"], "regression")
        self.assertEqual(self.uncertainty.data["rosenblatt"], True)


    def test_UQMC(self):
        self.setUpTestCalculations()

        self.uncertainty.UQ(method="mc", plot_condensed=False)

        self.assertEqual(self.uncertainty.data["function"], "MC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], ["a", "b"])


    def test_UQMCSingle(self):
        self.setUpTestCalculations()

        self.uncertainty.UQ(method="mc", plot_condensed=False, single=True)

        self.assertEqual(self.uncertainty.data["function"], "MC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], "b")



    def test_UQCustom(self):
        self.setUpTestCalculations()

        self.uncertainty.UQ(method="custom", custom_keyword="value")

        self.assertEqual(self.uncertainty.data["function"], "CustomUQ")
        self.assertEqual(self.uncertainty.data["custom_keyword"], "value")


    def test_CustomUQ(self):
        self.setUpTestCalculations()

        self.uncertainty.CustomUQ(custom_keyword="value")

        self.assertEqual(self.uncertainty.data["function"], "CustomUQ")
        self.assertEqual(self.uncertainty.data["custom_keyword"], "value")



    def compare_plot(self, name, compare_folder=""):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures", compare_folder,
                                    name + ".png")

        plot_file = os.path.join(self.output_test_dir, name + ".png")

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)


    def plot_exists(self, name):
        plot_file = os.path.join(self.output_test_dir, name + ".png")
        self.assertTrue(os.path.isfile(plot_file))


if __name__ == "__main__":
    unittest.main()
