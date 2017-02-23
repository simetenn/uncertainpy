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

from testing_modules import TestingFeatures
from models import TestingModel1d

from TestingUncertaintyCalculations import TestingUncertaintyCalculations

class TestUncertainty(unittest.TestCase):
    def setUp(self):
        # self.difference ="1e-8"
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


        uncertainty_calculations = UncertaintyCalculations(seed=self.seed, nr_mc_samples=10)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 uncertainty_calculations=uncertainty_calculations,
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error")



    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        UncertaintyEstimation(TestingModel1d())


    def test_intitFeatures(self):
        uncertainty = UncertaintyEstimation(TestingModel1d(),
                                            verbose_level="error")
        self.assertIsInstance(uncertainty.features, GeneralFeatures)

        uncertainty = UncertaintyEstimation(TestingModel1d(),
                                            features=TestingFeatures(),
                                            verbose_level="error")
        self.assertIsInstance(uncertainty.features, TestingFeatures)


    def test_initModel(self):
        uncertainty = UncertaintyEstimation(TestingModel1d(),
                                            verbose_level="error")
        self.assertIsInstance(uncertainty.model, TestingModel1d)


    def test_initUncertaintyCalculations(self):

        class TestingUncertaintyCalculations(UncertaintyCalculations):
            def PCECustom(self):
                "custom PCE method"

        uncertainty = UncertaintyEstimation(
            TestingModel1d(),
            uncertainty_calculations=TestingUncertaintyCalculations(TestingModel1d()),
            verbose_level="error"
        )

        self.assertIsInstance(uncertainty.uncertainty_calculations, TestingUncertaintyCalculations)


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
        result = subprocess.call(["h5diff", filename, compare_file])


        self.assertEqual(result, 0)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-b.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-b.h5")
        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)



    def test_PC(self):
        self.uncertainty.PC()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")
        self.assertTrue(os.path.isfile(filename))

        # TODO find out why this is needed for different machines
        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)



    def test_PCPlot(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

        model = TestingModel1d(parameters)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])


        uncertainty_calculations = UncertaintyCalculations(seed=self.seed, nr_mc_samples=10)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 uncertainty_calculations=uncertainty_calculations,
                                                 save_data=False,
                                                 save_figures=True,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error")


        self.uncertainty.PC()

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")


        self.compare_plot("total-sensitivity_1_grid")



    def test_PCSinglePlot(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

        model = TestingModel1d(parameters)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])


        uncertainty_calculations = UncertaintyCalculations(seed=self.seed, nr_mc_samples=10)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 uncertainty_calculations=uncertainty_calculations,
                                                 save_data=False,
                                                 save_figures=True,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error")


        self.uncertainty.PCSingle()

        self.compare_plot("TestingModel1d_single-parameter-a/directComparison_mean-variance",
                          compare_folder="")
        self.compare_plot("TestingModel1d_single-parameter-a/directComparison_confidence-interval",
                          compare_folder="")

        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_mean-variance",
                          compare_folder="")
        self.compare_plot("TestingModel1d_single-parameter-a/feature1d_confidence-interval",
                          compare_folder="")


    def test_MCSingle(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

        model = TestingModel1d(parameters)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        uncertainty_calculations = UncertaintyCalculations(seed=self.seed, nr_mc_samples=10)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 uncertainty_calculations=uncertainty_calculations,
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_data_filename="TestingModel1d_MC",
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error")


        self.uncertainty.MCSingle()


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC_single-parameter-a.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC_single-parameter-a.h5")

        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)



        compare_file = os.path.join(folder, "data/TestingModel1d_MC_single-parameter-b.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC_single-parameter-b.h5")

        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)



    def test_MC(self):

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

        model = TestingModel1d(parameters)

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        uncertainty_calculations = UncertaintyCalculations(seed=self.seed, nr_mc_samples=10**1)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 uncertainty_calculations=uncertainty_calculations,
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_data_filename="TestingModel1d_MC",
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error")


        self.uncertainty.MC()


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")
        self.assertTrue(os.path.isfile(filename))

        result = subprocess.call(["h5diff", filename, compare_file])

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

        # sys.exit(1)

        self.compare_plot("directComparison_mean")
        self.compare_plot("directComparison_variance")
        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")

        self.compare_plot("directComparison_sensitivity_1_a")
        self.compare_plot("directComparison_sensitivity_1_b")
        self.compare_plot("directComparison_sensitivity_1")
        self.compare_plot("directComparison_sensitivity_1_grid")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")
        self.compare_plot("feature1d_sensitivity_1")
        self.compare_plot("feature1d_sensitivity_1_grid")
        self.compare_plot("feature0d_total-sensitivity_1")

        self.compare_plot("directComparison_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")

        self.compare_plot("feature1d_sensitivity_t_a")
        self.compare_plot("feature1d_sensitivity_t_b")
        self.compare_plot("feature1d_sensitivity_t")
        self.compare_plot("feature1d_sensitivity_t_grid")



        self.compare_plot("directComparison_sensitivity_t_a")
        self.compare_plot("directComparison_sensitivity_t_b")
        self.compare_plot("directComparison_sensitivity_t")
        self.compare_plot("directComparison_sensitivity_t_grid")

        self.compare_plot("feature0d_total-sensitivity_t")


        self.compare_plot("directComparison_total-sensitivity_t")
        self.compare_plot("feature0d_total-sensitivity_t")
        self.compare_plot("feature1d_total-sensitivity_t")
        self.compare_plot("feature2d_total-sensitivity_t")



        self.compare_plot("total-sensitivity_t_grid")
        self.compare_plot("total-sensitivity_1_grid")


    def test_plotCondensed(self):
        self.uncertainty.PC()
        self.uncertainty.plot()

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")



    def test_plotNoSensitivity(self):
        self.uncertainty.PC()
        self.uncertainty.plot(condensed=False, sensitivity=False)

        self.compare_plot("directComparison_mean")
        self.compare_plot("directComparison_variance")
        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")



    def test_plotSimulatorResults(self):
        self.uncertainty.PC()
        self.uncertainty.plot(simulator_results=True)


        self.assertEqual(len(glob.glob(os.path.join(self.output_test_dir,
                                                    "simulator_results/*.png"))),
                         self.uncertainty.uncertainty_calculations.nr_pc_samples)



    def test_PCplotSimulatorResults(self):
        self.uncertainty.plot_simulator_results = True
        self.uncertainty.PC()

        self.assertEqual(len(glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png"))),
                         self.uncertainty.uncertainty_calculations.nr_pc_samples)



    def setUpTestCalculations(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        parameters.setAllDistributions(Distribution(0.5).uniform)

        model = TestingModel1d(parameters)

        features = TestingFeatures(features_to_run=None)


        uncertainty_calculations = TestingUncertaintyCalculations()

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 uncertainty_calculations=uncertainty_calculations,
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
        self.assertEqual(self.uncertainty.data["plot_condensed"], False)


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
        self.assertEqual(self.uncertainty.data["plot_condensed"], True)


    def test_UQMC(self):
        self.setUpTestCalculations()

        self.uncertainty.UQ(method="mc", plot_condensed=False)

        self.assertEqual(self.uncertainty.data["function"], "MC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], ["a", "b"])
        self.assertEqual(self.uncertainty.data["plot_condensed"], False)


    def test_UQMCSingle(self):
        self.setUpTestCalculations()

        self.uncertainty.UQ(method="mc", plot_condensed=False, single=True)

        self.assertEqual(self.uncertainty.data["function"], "MC")
        self.assertEqual(self.uncertainty.data["uncertain_parameters"], "b")
        self.assertEqual(self.uncertainty.data["plot_condensed"], False)



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



    def compare_plot(self, name, compare_folder="TestingModel1d"):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures", compare_folder,
                                    name + ".png")

        plot_file = os.path.join(self.output_test_dir, name + ".png")

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)




if __name__ == "__main__":
    unittest.main()
