import glob
import os
import unittest
import subprocess
import shutil
import matplotlib
import numpy as np

matplotlib.use('Agg')



from uncertainpy.plotting.plot_uncertainty import PlotUncertainty
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

        self.compare_plot("TestingModel1d_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")


    def test_total_sensitivity_all_t(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity_all(hardcopy=True, sensitivity="sensitivity_t")

        self.compare_plot("TestingModel1d_total-sensitivity_t")
        self.compare_plot("feature0d_total-sensitivity_t")
        self.compare_plot("feature1d_total-sensitivity_t")
        self.compare_plot("feature2d_total-sensitivity_t")


    def test_simulator_results_1d(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.plot.data = Data()

        self.plot.data.t["TestingModel1d"] = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.plot.data.U["TestingModel1d"] = [U, U, U, U, U]
        self.plot.data.features_1d = ["TestingModel1d"]
        self.plot.data.model_name = "TestingModel1d"
        self.plot.simulator_results_1d()

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
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


    def test_simulator_results_2d_error(self):
        self.plot.data = Data()

        with self.assertRaises(ValueError):
            self.plot.simulator_results_2d()


    def test_simulator_results_1d_model(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.plot.data = Data()

        self.plot.data.t["TestingModel1d"] = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.plot.data.U["TestingModel1d"] = [U, U, U, U, U]
        self.plot.data.features_1d = ["TestingModel1d"]
        self.plot.data.model_name = "TestingModel1d"
        self.plot.data.labels = {"TestingModel1d": ["x", "y"]}
        self.plot.simulator_results()

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 5)


    def test_simulator_results_2d_model(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel2d.h5"))

        self.plot.simulator_results_2d()

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 22)


    def test_simulator_results_2d(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel2d.h5"))

        self.plot.simulator_results()

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 22)



    def test_simulator_results_0d_model(self):
        self.plot.load(os.path.join(self.test_data_dir, "TestingModel0d.h5"))

        self.plot.simulator_results_0d()

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
        self.assertTrue(np.array_equal(self.plot.data.U["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.U["TestingModel1d"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.E["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.E["TestingModel1d"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.t["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.t["TestingModel1d"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.Var["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.Var["TestingModel1d"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.p_05["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.p_05["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.p_95["TestingModel1d"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.p_95["TestingModel1d"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.sensitivity_1["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.sensitivity_1["TestingModel1d"], [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_1["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_1["TestingModel1d"],
                                       [3., 4.]))

        self.assertTrue(np.array_equal(self.plot.data.sensitivity_t["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.sensitivity_t["TestingModel1d"],
                                       [3., 4.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_t["feature1d"],
                                       [1., 2.]))
        self.assertTrue(np.array_equal(self.plot.data.total_sensitivity_t["TestingModel1d"],
                                       [3., 4.]))


        self.assertEqual(self.plot.data.uncertain_parameters[0], "a")
        self.assertEqual(self.plot.data.uncertain_parameters[1], "b")

        self.assertEqual(self.plot.data.labels["TestingModel1d"], ["xlabel", "ylabel"])
        self.assertEqual(self.plot.data.labels["feature1d"], ["xlabel", "ylabel"])

        self.assertEqual(self.plot.data.feature_list[0], "TestingModel1d")
        self.assertEqual(self.plot.data.feature_list[1], "feature1d")



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

        self.plot.attribute_feature_1d(feature="TestingModel1d",
                                       attribute="E",
                                       attribute_name="mean")

        self.compare_plot("TestingModel1d_mean")


    def test_attribute_feature_1d_variance(self):
        self.plot.load(self.data_file_path)

        self.plot.attribute_feature_1d(feature="TestingModel1d",
                                       attribute="Var",
                                       attribute_name="variance")

        self.compare_plot("TestingModel1d_variance")



    def test_mean_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.mean_1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.mean_1d(feature="feature2d")


    def test_mean_model_result(self):
        self.plot.load(self.data_file_path)

        self.plot.mean_1d(feature="TestingModel1d")

        self.compare_plot("TestingModel1d_mean")


    def test_mean_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.mean_1d(feature="feature1d")

        self.compare_plot("feature1d_mean")


    def test_variance_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.variance_1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.variance_1d(feature="feature2d")


    def test_variance_model_result(self):
        self.plot.load(self.data_file_path)

        self.plot.variance_1d(feature="TestingModel1d")

        self.compare_plot("TestingModel1d_variance")


    def test_variance_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.variance_1d(feature="feature1d")

        self.compare_plot("feature1d_variance")



    def test_mean_variance_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.mean_variance_1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.mean_variance_1d(feature="feature2d")


    def test_mean_variance_model_result(self):
        self.plot.load(self.data_file_path)

        self.plot.mean_variance_1d(feature="TestingModel1d")

        self.compare_plot("TestingModel1d_mean-variance")


    def test_mean_variance_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.mean_variance_1d(feature="feature1d")

        self.compare_plot("feature1d_mean-variance")



    def test_confidence_interval_model_result(self):
        self.plot.load(self.data_file_path)

        self.plot.confidence_interval_1d(feature="TestingModel1d")

        self.compare_plot("TestingModel1d_confidence-interval")


    def test_confidence_interval_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.confidence_interval_1d(feature="feature1d")

        self.compare_plot("feature1d_confidence-interval")


    def test_confidence_interval_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.confidence_interval_1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.confidence_interval_1d(feature="feature2d")


    def test_sensitivity_1_model_result(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d(feature="TestingModel1d", sensitivity="sensitivity_1")

        self.compare_plot("TestingModel1d_sensitivity_1_a")
        self.compare_plot("TestingModel1d_sensitivity_1_b")


    def test_sensitivity_model_result_t(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d(feature="TestingModel1d", sensitivity="sensitivity_t")

        self.compare_plot("TestingModel1d_sensitivity_t_a")
        self.compare_plot("TestingModel1d_sensitivity_t_b")


    def test_sensitivity_1_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d(feature="feature1d", sensitivity="sensitivity_1")


        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")


    def test_sensitivity_t_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d(feature="feature1d", sensitivity="sensitivity_t")


        self.compare_plot("feature1d_sensitivity_t_a")
        self.compare_plot("feature1d_sensitivity_t_b")


    def test_sensitivity_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.sensitivity_1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.sensitivity_1d(feature="feature2d")



    def test_sensitivity_1_combined_model_result(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d_combined(feature="TestingModel1d", sensitivity="sensitivity_1")

        self.compare_plot("TestingModel1d_sensitivity_1")


    def test_sensitivity_t_combined_model_result(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d_combined(feature="TestingModel1d", sensitivity="sensitivity_t")

        self.compare_plot("TestingModel1d_sensitivity_t")


    def test_sensitivity_1_combined_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d_combined(feature="feature1d", sensitivity="sensitivity_1")

        self.compare_plot("feature1d_sensitivity_1")

    def test_sensitivity_t_combined_feature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.sensitivity_1d_combined(feature="feature1d", sensitivity="sensitivity_t")

        self.compare_plot("feature1d_sensitivity_t")


    def test_sensitivity_combined_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.sensitivity_1d_combined(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.sensitivity_1d_combined(feature="feature2d")



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


    def test_features_1d_no_t(self):
        self.plot.load(self.data_file_path)
        self.plot.data.t["feature1d"] = np.nan

        self.plot.features_1d(sensitivity="sensitivity_1")

        self.plot_exists("feature1d_mean")
        self.plot_exists("feature1d_variance")
        self.plot_exists("feature1d_mean-variance")
        self.plot_exists("feature1d_confidence-interval")
        self.plot_exists("feature1d_sensitivity_1_a")
        self.plot_exists("feature1d_sensitivity_1_b")
        self.plot_exists("feature1d_sensitivity_1")



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



    def test_features_0d_t(self):
        self.plot.load(self.data_file_path)

        self.plot.feature_0d(feature="feature0d", sensitivity="sensitivity_t")

        self.compare_plot("feature0d_sensitivity_t")


    def test_plot_condensed(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_condensed(sensitivity="sensitivity_1")

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_plot_all_1(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_all("sensitivity_1")

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


        self.compare_plot("total-sensitivity_1_grid")

    def test_plot_all_t(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_all("sensitivity_t")


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


    def test_plot_all_sensitivities(self):
        self.plot.load(self.data_file_path)

        self.plot.plot_all_sensitivities()

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

        self.compare_plot("total-sensitivity_1_grid")


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



    def test_plot_condensed_sensitivity_1(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=True, sensitivity="sensitivity_1")

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")
        self.compare_plot("TestingModel1d_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_plot_condensed_no_sensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=True, sensitivity=None)

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature0d")



    def test_plot_all_no_sensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=False, sensitivity=None)

        self.compare_plot("TestingModel1d_mean")
        self.compare_plot("TestingModel1d_variance")
        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")



    def test_plot_all(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=False, sensitivity="sensitivity_1")

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

        self.compare_plot("TestingModel1d_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


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



    def compare_plot(self, name):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures",
                                    name + self.figureformat)

        plot_file = os.path.join(self.output_test_dir, name + self.figureformat)

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)


    def plot_exists(self, name):
        plot_file = os.path.join(self.output_test_dir, name + self.figureformat)
        self.assertTrue(os.path.isfile(plot_file))

# TODO test combined features 0 for many features

# TODO test plot_allFromExploration
# TODO test plot_folder



if __name__ == "__main__":
    unittest.main()
