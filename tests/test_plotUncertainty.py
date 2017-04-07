import numpy as np
import glob
import os
import unittest
import subprocess
import shutil
import matplotlib

matplotlib.use('Agg')



from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy import Data


class TestPlotUncertainpy(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.test_data_dir = os.path.join(self.folder, "data")
        self.data_file = "TestingModel1d.h5"
        self.data_file_path = os.path.join(self.test_data_dir, self.data_file)
        self.output_test_dir = ".tests/"

        self.figureformat = ".png"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.plot = PlotUncertainty(output_dir=self.output_test_dir,
                                    verbose_level="warning",
                                    figureformat=self.figureformat)




    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)



    def test_total_sensitivity_grid_1(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity_grid(hardcopy=True, sensitivity="sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_total_sensitivity_grid_t(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity_grid(hardcopy=True, sensitivity="sensitivity_t")

        self.compare_plot("total-sensitivity_t_grid")


    def test_total_sensitivity_1(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity(feature="feature1d",
                                    sensitivity="sensitivity_1",
                                    hardcopy=True)

        self.compare_plot("feature1d_total-sensitivity_1")

    def test_total_sensitivity_t(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity(feature="feature1d",
                                    sensitivity="sensitivity_t",
                                    hardcopy=True)

        self.compare_plot("feature1d_total-sensitivity_t")


    def test_total_sensitivity_all_1(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity_all(hardcopy=True, sensitivity="sensitivity_1")

        self.compare_plot("directComparison_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")


    def test_total_sensitivity_all_t(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity_all(hardcopy=True, sensitivity="sensitivity_t")

        self.compare_plot("directComparison_total-sensitivity_t")
        self.compare_plot("feature0d_total-sensitivity_t")
        self.compare_plot("feature1d_total-sensitivity_t")
        self.compare_plot("feature2d_total-sensitivity_t")


    def test_simulator_results_1d(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.plot.data = Data()

        self.plot.data.t["directComparison"] = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.plot.data.U["directComparison"] = [U, U, U, U, U]
        self.plot.data.features_1d = ["directComparison"]
        self.plot.simulator_results_1d()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures/U.png")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            result = subprocess.call(["diff", plot, compare_file])

            self.assertEqual(result, 0)

            plot_count += 1

        self.assertEqual(plot_count, 5)



    def test_simulator_results_0d_error(self):
        self.plot.data = Data()

        with self.assertRaises(ValueError):
            self.plot.simulator_results_1d()

    def test_simulator_results_1d_error(self):
        self.plot.data = Data()

        with self.assertRaises(ValueError):
            self.plot.simulator_results_1d()

    def test_simulator_results_error(self):
        self.plot.data = Data()

        with self.assertRaises(NotImplementedError):
            self.plot.simulator_results()


    def test_simulator_results_1d_model(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.plot.data = Data()

        self.plot.data.t["directComparison"] = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.plot.data.U["directComparison"] = [U, U, U, U, U]
        self.plot.data.features_1d = ["directComparison"]
        self.plot.simulator_results()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures/U.png")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            result = subprocess.call(["diff", plot, compare_file])

            self.assertEqual(result, 0)

            plot_count += 1

        self.assertEqual(plot_count, 5)


    def test_simulator_results_2d(self):
        self.plot.data = Data()

        with self.assertRaises(NotImplementedError):
            self.plot.simulator_results()


    def test_simulator_results_0d_model(self):
        self.plot.load(os.path.join(self.test_data_dir, "TestingModel0d.h5"))

        self.plot.simulator_results()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures", "simulator_results", "U" + self.figureformat)

        plot_file = os.path.join(self.output_test_dir, "simulator_results", "U" + self.figureformat)

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)


    def test_simulator_results_0d(self):
        self.plot.load(os.path.join(self.test_data_dir, "TestingModel0d.h5"))

        self.plot.simulator_results()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures", "simulator_results", "U" + self.figureformat)

        plot_file = os.path.join(self.output_test_dir, "simulator_results", "U" + self.figureformat)

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)



    def test_init(self):
        plot = PlotUncertainty(output_dir=self.output_test_dir,
                               verbose_level="error")

        self.assertIsInstance(plot, PlotUncertainty)


    def test_load(self):
        self.plot.load(os.path.join(self.test_data_dir, "test_save_mock"))

        self.assert_data()


    def test_set_data(self):
        data = Data()

        data.load(os.path.join(self.test_data_dir, "test_save_mock"))

        self.plot.data = data

        self.assert_data()

    def test_set_output_dir(self):
        test_dir = os.path.join(self.output_test_dir, "testing")
        self.plot.output_dir = test_dir

        self.assertTrue(os.path.isdir)


    def assert_data(self):
        self.assertTrue(np.array_equal(self.plot.data.U["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.U["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.E["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.E["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.t["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.t["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.Var["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.Var["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.p_05["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.p_05["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.p_95["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.p_95["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.sensitivity_1["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.sensitivity_1["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_1["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_1["directComparison"],
                                       [3., 4.]))

        self.assertTrue(np.array_equal(self.plot.data.sensitivity_t["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.sensitivity_t["directComparison"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_t["feature1"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_t["directComparison"],
                                       [3., 4.]))


        self.assertEqual(self.plot.data.uncertain_parameters[0], "a")
        self.assertEqual(self.plot.data.uncertain_parameters[1], "b")

        self.assertEqual(self.plot.data.xlabel, "xlabel")
        self.assertEqual(self.plot.data.ylabel, "ylabel")

        self.assertEqual(self.plot.data.feature_list[0], "directComparison")
        self.assertEqual(self.plot.data.feature_list[1], "feature1")



    def test_attribute_feature_1d_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_1d(feature="feature2d")

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_1d(attribute="test")


    def test_attribute_feature_1d_mean(self):
        self.plot.load(self.data_file_path)

        self.plot.attribute_feature_1d(feature="directComparison",
                                       attribute="E",
                                       attribute_name="mean")

        self.compare_plot("directComparison_mean")


    def test_attribute_feature_1d_variance(self):
        self.plot.load(self.data_file_path)

        self.plot.attribute_feature_1d(feature="directComparison",
                                       attribute="Var",
                                       attribute_name="variance")

        self.compare_plot("directComparison_variance")



    def test_mean_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.mean(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.mean(feature="feature2d")


    def test_mean_DirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.mean(feature="directComparison")

        self.compare_plot("directComparison_mean")


    def test_mean_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.mean(feature="feature1d")

        self.compare_plot("feature1d_mean")


    def test_variance_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.variance(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.variance(feature="feature2d")


    def test_variance_DirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.variance(feature="directComparison")

        self.compare_plot("directComparison_variance")


    def test_variance_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.variance(feature="feature1d")

        self.compare_plot("feature1d_variance")



    def test_mean_variance_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.mean_variance(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.mean_variance(feature="feature2d")


    def test_mean_variance_DirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.mean_variance(feature="directComparison")

        self.compare_plot("directComparison_mean-variance")


    def test_mean_variance_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.mean_variance(feature="feature1d")

        self.compare_plot("feature1d_mean-variance")



    def test_confidence_interval_DirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.confidence_interval(feature="directComparison")

        self.compare_plot("directComparison_confidence-interval")


    def test_confidence_interval_feature0d(self):
        self.plot.load(self.data_file_path)

        self.plot.confidence_interval(feature="feature1d")

        self.compare_plot("feature1d_confidence-interval")


    def test_confidence_interval_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.confidence_interval(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.confidence_interval(feature="feature2d")


    def test_sensitivity_DirectComparison_1(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity(feature="directComparison", sensitivity="sensitivity_1")

        self.compare_plot("directComparison_sensitivity_1_a")
        self.compare_plot("directComparison_sensitivity_1_b")


    def test_sensitivity_DirectComparisonT(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity(feature="directComparison", sensitivity="sensitivity_t")

        self.compare_plot("directComparison_sensitivity_t_a")
        self.compare_plot("directComparison_sensitivity_t_b")


    def test_sensitivityFeature1d1(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity(feature="feature1d", sensitivity="sensitivity_1")


        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")


    def test_sensitivityFeature1dt(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity(feature="feature1d", sensitivity="sensitivity_t")


        self.compare_plot("feature1d_sensitivity_t_a")
        self.compare_plot("feature1d_sensitivity_t_b")


    def test_sensitivityError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.sensitivity(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.sensitivity(feature="feature2d")



    def test_sensitivity_combinedDirectComparison1(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_combined(feature="directComparison", sensitivity="sensitivity_1")

        self.compare_plot("directComparison_sensitivity_1")


    def test_sensitivity_combinedDirectComparisonT(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_combined(feature="directComparison", sensitivity="sensitivity_t")

        self.compare_plot("directComparison_sensitivity_t")


    def test_sensitivity_combinedFeature1d1(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_combined(feature="feature1d", sensitivity="sensitivity_1")

        self.compare_plot("feature1d_sensitivity_1")

    def test_sensitivity_combinedFeature1dT(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_combined(feature="feature1d", sensitivity="sensitivity_t")

        self.compare_plot("feature1d_sensitivity_t")


    def test_sensitivity_combinedError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.sensitivity_combined(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.sensitivity_combined(feature="feature2d")



    def test_features_1d(self):
        self.plot.load(self.data_file_path)

        self.plot.features_1d(sensitivity="sensitivity_1")

        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")
        self.compare_plot("feature1d_sensitivity_1")



    def test_feature_0d(self):
        self.plot.load(self.data_file_path)

        self.plot.feature_0d(feature="feature0d", sensitivity="sensitivity_1")

        self.compare_plot("feature0d_sensitivity_1")

        self.plot.feature_0d(feature="feature0d", sensitivity="sensitivity_t")

        self.compare_plot("feature0d_sensitivity_t")


        with self.assertRaises(ValueError):
            self.plot.feature_0d(feature="feature1d")


    def test_features_0d1(self):
        self.plot.load(self.data_file_path)

        self.plot.feature_0d(feature="feature0d", sensitivity="sensitivity_1")

        self.compare_plot("feature0d_sensitivity_1")



    def test_features_0dT(self):
        self.plot.load(self.data_file_path)

        self.plot.feature_0d(feature="feature0d", sensitivity="sensitivity_t")

        self.compare_plot("feature0d_sensitivity_t")


    def test_plot_condensed(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_condensed(sensitivity="sensitivity_1")

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")



    def test_plot_condensed_no_sensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_condensed(sensitivity=None)

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature0d")



    def test_plot_all1(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_all("sensitivity_1")

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


        self.compare_plot("total-sensitivity_1_grid")

    def test_plot_allT(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_all("sensitivity_t")


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


    def test_plot_allNoSensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_all(sensitivity=None)

        self.compare_plot("directComparison_mean")
        self.compare_plot("directComparison_variance")
        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")




    def test_plot_all_sensitivities(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_all_sensitivities()

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

        self.compare_plot("total-sensitivity_1_grid")


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



    def test_plot_condensed_sensitivity_1(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=True, sensitivity="sensitivity_1")

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_plot_condensed_no_sensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=True, sensitivity=None)

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature0d")



    def test_plot_all_no_sensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=False, sensitivity=None)

        self.compare_plot("directComparison_mean")
        self.compare_plot("directComparison_variance")
        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")



    def test_plotall(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=False, sensitivity="sensitivity_1")

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

        self.compare_plot("directComparison_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


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



    def compare_plot(self, name):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures",
                                    name + self.figureformat)

        plot_file = os.path.join(self.output_test_dir, name + self.figureformat)

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)

# TODO test combined features 0 for many features

# TODO test plot_allFromExploration
# TODO test plot_folder



if __name__ == "__main__":
    unittest.main()
