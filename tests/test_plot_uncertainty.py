import glob
import os
import unittest
import subprocess
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')

from .testing_classes import TestCasePlot
from uncertainpy.plotting.plot_uncertainty import PlotUncertainty
from uncertainpy import Data



class TestPlotUncertainpy(TestCasePlot):
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

        self.data_types = ["U", "t", "mean", "variance", "percentile_5", "percentile_95",
                           "sensitivity_1", "sensitivity_1_sum",
                           "sensitivity_t", "sensitivity_t_sum"]


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


    def test_sensitivity_1_sum(self):
        self.plot.load(self.data_file_path)

        self.plot.total_sensitivity(feature="feature1d",
                                    sensitivity="sensitivity_1",
                                    hardcopy=True)

        self.compare_plot("feature1d_total-sensitivity_1")


    def test_sensitivity_t_sum(self):
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


    def test_results_1d(self):

        self.plot.data = Data()

        self.plot.data.add_features("TestingModel1d")
        self.plot.data["TestingModel1d"].labels = ["x", "y"]
        self.plot.data["TestingModel1d"]["t"] = np.load(os.path.join(self.folder, "data/t_test.npy"))
        values = np.load(os.path.join(self.folder, "data/U_test.npy"))
        self.plot.data["TestingModel1d"]["U"] = [U, U, U, U, U]
        self.plot.data.model_name = "TestingModel1d"

        self.plot.results_1d(feature="TestingModel1d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "TestingModel1d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 5)


    def test_results_0d_error(self):
        with self.assertRaises(ValueError):
            self.plot.results_0d(feature="TestingModel1d")

        self.plot.data = Data()

        self.plot.results_0d(feature="TestingModel1d")

        self.plot.data.add_features("TestingModel1d")
        self.plot.data["TestingModel1d"].labels = ["x", "y"]
        self.plot.data["TestingModel1d"]["t"] = np.load(os.path.join(self.folder, "data/t_test.npy"))
        values = np.load(os.path.join(self.folder, "data/U_test.npy"))
        self.plot.data["TestingModel1d"]["U"] = [U, U, U, U, U]
        self.plot.data.model_name = "TestingModel1d"

        with self.assertRaises(ValueError):
            self.plot.results_0d(feature="TestingModel1d")


    def test_results_1d_error(self):
        with self.assertRaises(ValueError):
            self.plot.results_1d(feature="TestingModel1d")

        self.plot.data = Data()

        self.plot.results_1d(feature="TestingModel1d")

        self.plot.data.add_features("TestingModel1d")
        self.plot.data["TestingModel1d"]["U"] = [1, 1, 1, 1, 1]
        self.plot.data.model_name = "TestingModel1d"

        with self.assertRaises(ValueError):
            self.plot.results_1d(feature="TestingModel1d")


    def test_results_2d_error(self):
        with self.assertRaises(ValueError):
            self.plot.results_2d(feature="TestingModel1d")

        self.plot.data = Data()

        self.plot.results_2d(feature="TestingModel1d")

        self.plot.data.add_features("TestingModel1d")
        self.plot.data["TestingModel1d"].labels = ["x", "y"]
        self.plot.data["TestingModel1d"]["t"] = np.load(os.path.join(self.folder, "data/t_test.npy"))
        values = np.load(os.path.join(self.folder, "data/U_test.npy"))
        self.plot.data["TestingModel1d"]["U"] = [U, U, U, U, U]
        self.plot.data.model_name = "TestingModel1d"

        with self.assertRaises(ValueError):
            self.plot.results_2d(feature="TestingModel1d")


    def test_results_1d_model(self):

        self.plot.data = Data()


        self.plot.data.add_features("TestingModel1d")
        self.plot.data["TestingModel1d"].labels = ["x", "y"]
        self.plot.data["TestingModel1d"]["t"] = np.load(os.path.join(self.folder, "data/t_test.npy"))
        values = np.load(os.path.join(self.folder, "data/U_test.npy"))
        self.plot.data["TestingModel1d"]["U"] = [U, U, U, U, U]
        self.plot.data.model_name = "TestingModel1d"

        self.plot.results(feature="TestingModel1d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "TestingModel1d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 5)



    def test_results(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel1d.h5"))

        self.plot.all_results()

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature0d_results/*.png")))
        self.assertEqual(plot_count, 1)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature1d_results/*.png")))
        self.assertEqual(plot_count, 21)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/feature2d_results/*.png")))
        self.assertEqual(plot_count, 21)

        plot_count = len(glob.glob(os.path.join(self.output_test_dir, "results/TestingModel1d_results/*.png")))
        self.assertEqual(plot_count, 21)

    def test_results_2d_feature(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel1d.h5"))

        self.plot.results_2d(feature="feature2d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "feature2d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 21)


    def test_results_feature_2d(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel1d.h5"))

        self.plot.results(feature="feature2d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "feature2d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 21)


    def test_results_0d_feature(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel1d.h5"))

        self.plot.results_0d(feature="feature0d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "feature0d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 1)


    def test_results_feature_0d(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel1d.h5"))

        self.plot.results(feature="feature0d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "feature0d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 1)


    def test_results_1d_feature(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel1d.h5"))

        self.plot.results_1d(feature="feature1d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "feature1d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 21)


    def test_results_feature_1d(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel1d.h5"))

        self.plot.results(feature="feature1d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "feature1d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 21)


    def test_results_2d_model(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel2d.h5"))

        self.plot.results_2d(feature="TestingModel2d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "TestingModel2d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 21)


    def test_results_2d(self):
        self.plot.data = Data(os.path.join(self.test_data_dir, "TestingModel2d.h5"))

        self.plot.results(feature="TestingModel2d")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "TestingModel2d_results/*.png")):
            plot_count += 1

        self.assertEqual(plot_count, 21)



    def test_results_0d_model(self):
        self.plot.load(os.path.join(self.test_data_dir, "TestingModel0d.h5"))

        self.plot.results_0d(feature="TestingModel0d")

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures", "results", "U" + self.figureformat)

        plot_file = os.path.join(self.output_test_dir, "TestingModel0d_results", "U" + self.figureformat)

        if self.exact_plots:
            result = subprocess.call(["diff", plot_file, compare_file])
            self.assertEqual(result, 0)
        else:
            self.assertTrue(os.path.isfile(plot_file))


    def test_results_0d(self):
        self.plot.load(os.path.join(self.test_data_dir, "TestingModel0d.h5"))

        self.plot.results(feature="TestingModel0d")

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures", "results", "U" + self.figureformat)

        plot_file = os.path.join(self.output_test_dir, "TestingModel0d_results", "U" + self.figureformat)

        if self.exact_plots:
            result = subprocess.call(["diff", plot_file, compare_file])
            self.assertEqual(result, 0)
        else:
            self.assertTrue(os.path.isfile(plot_file))



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


        for data_type in self.data_types:
            self.assertTrue(np.array_equal(self.plot.data["feature1d"][data_type], [1., 2.]))
            self.assertTrue(np.array_equal(self.plot.data["TestingModel1d"][data_type], [3., 4.]))

        self.assertEqual(self.plot.data.uncertain_parameters, ["a", "b"])

        self.assertTrue(np.array_equal(self.plot.data["TestingModel1d"]["labels"], ["xlabel", "ylabel"]))
        self.assertTrue(np.array_equal(self.plot.data["feature1d"]["labels"], ["xlabel", "ylabel"]))




    def test_attribute_feature_1d_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_1d(feature="feature2d")

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_1d(attribute="test")


    def test_attribute_feature_2d_error(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_2d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_2d(feature="feature1d")

        with self.assertRaises(ValueError):
            self.plot.attribute_feature_2d(attribute="test")


    def test_attribute_feature_1d_mean(self):
        self.plot.load(self.data_file_path)

        self.plot.attribute_feature_1d(feature="TestingModel1d",
                                       attribute="mean",
                                       attribute_name="mean")

        self.compare_plot("TestingModel1d_mean")


    def test_attribute_feature_1d_variance(self):
        self.plot.load(self.data_file_path)

        self.plot.attribute_feature_1d(feature="TestingModel1d",
                                       attribute="variance",
                                       attribute_name="variance")

        self.compare_plot("TestingModel1d_variance")



    def test_attribute_feature_2d_mean(self):
        self.plot.load(self.data_file_path)

        self.plot.attribute_feature_2d(feature="feature2d",
                                       attribute="mean",
                                       attribute_name="mean")

        self.compare_plot("feature2d_mean")


    def test_attribute_feature_2d_variance(self):
        self.plot.load(self.data_file_path)

        self.plot.attribute_feature_2d(feature="feature2d",
                                       attribute="variance",
                                       attribute_name="variance")

        self.compare_plot("feature2d_variance")


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


    def test_mean_2d(self):
        self.plot.load(self.data_file_path)

        self.plot.mean_2d("feature2d")

        self.compare_plot("feature2d_mean")


    def test_variance_2d(self):
        self.plot.load(self.data_file_path)

        self.plot.variance_2d("feature2d")

        self.compare_plot("feature2d_variance")


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


    def test_feature_2d(self):
        self.plot.load(self.data_file_path)

        self.plot.features_2d()

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")


    def test_features_1d_no_t(self):
        self.plot.load(self.data_file_path)
        del self.plot.data["feature1d"]["t"]

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

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")



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


        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")


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

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")


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

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")


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

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")


    def test_plot_condensed_no_sensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=True, sensitivity=None)

        self.compare_plot("TestingModel1d_mean-variance")
        self.compare_plot("TestingModel1d_confidence-interval")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature0d")

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")



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

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")



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

        self.compare_plot("feature2d_mean")
        self.compare_plot("feature2d_variance")


# TODO test combined features 0 for many features

# TODO test plot_folder



if __name__ == "__main__":
    unittest.main()
