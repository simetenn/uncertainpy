import numpy as np
import os
import unittest
import subprocess
import shutil


from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.features import TestingFeatures
from uncertainpy.models import TestingModel1d

class TestPlotUncertainpy(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.test_data_dir = os.path.join(self.folder, "data")
        self.data_file = "test_plot_data"
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.plot = PlotUncertainty(data_dir=self.test_data_dir,
                                    output_dir_figures=self.output_test_dir,
                                    output_dir_gif=self.output_test_dir,
                                    verbose_level="error")



#     def tearDown(self):
#         if os.path.isdir(self.output_test_dir):
#             shutil.rmtree(self.output_test_dir)
#
#
#     def test_init(self):
#         plot = PlotUncertainty(data_dir=self.test_data_dir,
#                                output_dir_figures=self.output_test_dir,
#                                output_dir_gif=self.output_test_dir,
#                                verbose_level="error")
#
#         self.assertIsInstance(plot, PlotUncertainty)
#
#
#     def test_loadData(self):
#         self.plot.loadData("test_save_data")
#
#         self.assertData()
#
#
#     def test_setData(self):
#         self.data = PlotUncertainty(data_dir=self.test_data_dir,
#                                     output_dir_figures=self.output_test_dir,
#                                     output_dir_gif=self.output_test_dir,
#                                     verbose_level="error")
#
#
#         self.data.loadData("test_save_data")
#
#
#         self.plot.setData(t=self.data.t,
#                           U=self.data.U,
#                           E=self.data.E,
#                           Var=self.data.Var,
#                           p_05=self.data.p_05,
#                           p_95=self.data.p_95,
#                           uncertain_parameters=self.data.uncertain_parameters,
#                           sensitivity=self.data.sensitivity)
#
#         self.assertData()
#
#
#     def assertData(self):
#         model = TestingModel1d()
#         model.run()
#         t = model.t
#         U = model.U
#
#         feature = TestingFeatures()
#
#         self.assertTrue(np.array_equal(self.plot.t["directComparison"], t))
#         self.assertTrue(np.array_equal(self.plot.t["feature1d"], t))
#
#
#         self.assertTrue(self.plot.U["directComparison"].shape, (10, 21))
#         self.assertTrue(self.plot.U["feature1d"].shape, (10, 21))
#
#         self.assertTrue(np.allclose(self.plot.E["directComparison"], U, atol=0.001))
#         self.assertTrue(np.allclose(self.plot.E["feature1d"], feature.feature1d(), atol=0.001))
#
#         self.assertTrue(np.allclose(self.plot.Var["directComparison"], np.zeros(10) + 0.1, atol=0.01))
#         self.assertTrue(np.allclose(self.plot.Var["feature1d"], np.zeros(10), atol=0.001))
#
#
#         self.assertTrue(np.all(np.less(self.plot.p_05["directComparison"], U)))
#         self.assertTrue(np.allclose(self.plot.p_05["feature1d"], feature.feature1d(), atol=0.001))
#
#         self.assertTrue(np.all(np.greater(self.plot.p_95["directComparison"], U)))
#         self.assertTrue(np.allclose(self.plot.p_95["feature1d"], feature.feature1d(), atol=0.001))
#
#         self.assertTrue(self.plot.sensitivity["directComparison"].shape, (10, 2))
#         self.assertTrue(self.plot.sensitivity["feature1d"].shape, (10, 2))
#
#         self.assertEqual(len(self.plot.features_0d), 0)
#         self.assertEqual(len(self.plot.features_1d), 2)
#
#         self.assertEqual(len(self.plot.uncertain_parameters), 2)
#         self.assertTrue(self.plot.loaded_flag)
#
#
#
#     def test_sortFeatures(self):
#         logfile = os.path.join(self.output_test_dir, "test.log")
#
#
#         self.plot = PlotUncertainty(data_dir=self.test_data_dir,
#                                     output_dir_figures=self.output_test_dir,
#                                     output_dir_gif=self.output_test_dir,
#                                     verbose_level="warning",
#                                     verbose_filename=logfile)
#
#         self.plot.loadData(self.data_file)
#
#         features_0d, features_1d = self.plot.sortFeatures(self.plot.E)
#
#         test_file_content = """WARNING - plotUncertainty - No support for more than 0d and 1d plotting.
# WARNING - plotUncertainty - No support for more than 0d and 1d plotting."""
#
#         self.assertTrue(test_file_content in open(logfile).read())
#         self.assertEqual(features_0d, ["feature0d"])
#         self.assertEqual(features_1d, ["directComparison", "feature1d"])
#
#
#
#     def test_plotMean(self):
#         self.plot.loadData(self.data_file)
#
#         self.compare_plotType("plotMean", "mean", "directComparison")
#         self.compare_plotType("plotMean", "mean", "feature1d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotMean(feature="feature0d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotMean(feature="feature2d")
#
#
#
#     def test_plotAttributeFeature1d(self):
#         self.plot.loadData(self.data_file)
#
#         with self.assertRaises(ValueError):
#             self.plot.plotAttributeFeature1d(feature="feature0d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotAttributeFeature1d(feature="feature2d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotAttributeFeature1d(attribute="test")
#
#         self.compare_plotType("plotAttributeFeature1d", "mean", "directComparison",
#                               attribute="E", attribute_name="mean")
#         self.compare_plotType("plotAttributeFeature1d", "mean", "directComparison",
#                               attribute="Var", attribute_name="variance")
#
#         self.compare_plotType("plotAttributeFeature1d", "mean", "feature1d",
#                               attribute="E", attribute_name="mean")
#         self.compare_plotType("plotAttributeFeature1d", "mean", "feature1d",
#                               attribute="Var", attribute_name="variance")
#
#
#
# #     def test_plotVariance(self):
#         self.plot.loadData(self.data_file)
#
#
#         self.compare_plotType("plotVariance", "variance", "directComparison")
#         self.compare_plotType("plotVariance", "variance", "feature1d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotVariance(feature="feature0d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotVariance(feature="feature2d")
#
#
#
#     def test_plotMeanAndVariance(self):
#         self.plot.loadData(self.data_file)
#
#         self.compare_plotType("plotMeanAndVariance", "mean-variance", "directComparison")
#         self.compare_plotType("plotMeanAndVariance", "mean-variance", "feature1d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotMeanAndVariance(feature="feature0d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotMeanAndVariance(feature="feature2d")
#
#
#
#     def test_plotConfidenceInterval(self):
#         self.plot.loadData(self.data_file)
#
#         self.compare_plotType("plotConfidenceInterval", "confidence-interval", "directComparison")
#         self.compare_plotType("plotConfidenceInterval", "confidence-interval", "feature1d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotConfidenceInterval(feature="feature0d")
#
#         with self.assertRaises(ValueError):
#             self.plot.plotConfidenceInterval(feature="feature2d")


    def test_plotSensitivityDirectComparison(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivity(feature="directComparison")

        self.compare_plot("directComparison_sensitivity_a")
        self.compare_plot("directComparison_sensitivity_b")



    def test_plotSensitivityFeature1d(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivity(feature="feature1d")


        self.compare_plot("feature1d_sensitivity_a")
        self.compare_plot("feature1d_sensitivity_b")


    def test_plotSensitivityError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotSensitivity(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotSensitivity(feature="feature2d")



    def test_plotSensitivityCombinedDirectComparison(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivityCombined(feature="directComparison")

        self.compare_plot("directComparison_sensitivity")


    def test_plotSensitivityCombinedFeature1d(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivityCombined(feature="feature1d")

        self.compare_plot("feature1d_sensitivity")


    def test_plotSensitivityCombinedError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotSensitivityCombined(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotSensitivityCombined(feature="feature2d")



    def compare_plotType(self, plot_type, name, feature, **kwargs):
        getattr(self.plot, plot_type)(feature=feature, **kwargs)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_plot_data_figures",
                                    feature + "_" + name + ".png")

        plot_file = os.path.join(self.output_test_dir, self.data_file,
                                 feature + "_" + name + ".png")
        result = subprocess.call(["diff", plot_file, compare_file])

        self.assertEqual(result, 0)


    # def test_plot1dFeatures(self):
    #     self.plot.loadData(self.data_file)
    #
    #     self.plot.plot1dFeatures()
    #
    #
    #     self.compare_plot("feature1d_mean")
    #     self.compare_plot("feature1d_variance")
    #     self.compare_plot("feature1d_mean-variance")
    #     self.compare_plot("feature1d_confidence-interval")
    #     self.compare_plot("feature1d_sensitivity_a")
    #     self.compare_plot("feature1d_sensitivity_b")
    #     self.compare_plot("feature1d_sensitivity")


    def compare_plot(self, name, compare=False):
        folder = os.path.dirname(os.path.realpath(__file__))
        if compare:
            compare_file = os.path.join(folder, "data/compare",
                                        name + ".png")
            plot_file = os.path.join(self.output_test_dir,
                                     "compare", name + ".png")

        else:
            compare_file = os.path.join(folder, "data/test_plot_data_figures",
                                        name + ".png")

            plot_file = os.path.join(self.output_test_dir, self.data_file,
                                     name + ".png")

        result = subprocess.call(["diff", plot_file, compare_file])

        self.assertEqual(result, 0)

    #
    # def test_plot0dFeature(self):
    #     self.plot.loadData(self.data_file)
    #
    #     self.plot.plot0dFeature(feature="feature0d")
    #
    #     self.compare_plot("feature0d")
    #
    #     with self.assertRaises(ValueError):
    #         self.plot.plot0dFeature(feature="feature1d")
    #
    #
    # def test_plot0dFeatures(self):
    #     self.plot.loadData(self.data_file)
    #
    #     self.plot.plot0dFeature(feature="feature0d")
    #
    #     self.compare_plot("feature0d")
    #
    #
    #
    # def test_plotAllData(self):
    #     self.plot.loadData(self.data_file)
    #
    #     self.plot.plotAllData()
    #
    #
    #     self.compare_plot("feature1d_mean")
    #     self.compare_plot("feature1d_variance")
    #     self.compare_plot("feature1d_mean-variance")
    #     self.compare_plot("feature1d_confidence-interval")
    #     self.compare_plot("feature1d_sensitivity_a")
    #     self.compare_plot("feature1d_sensitivity_b")
    #     self.compare_plot("feature1d_sensitivity")
    #     self.compare_plot("feature1d_sensitivity_grid")
    #     self.compare_plot("feature0d")
    #
    #
    # def test_loadCompareData(self):
    #     self.plot.loadCompareData("TestingModel1d", ["pc", "mc_10", "mc_100"])
    #
    #     model = TestingModel1d()
    #     model.run()
    #     t = model.t
    #     U = model.U
    #
    #     feature = TestingFeatures()
    #
    #     self.assertTrue(np.array_equal(self.plot.t["directComparison"], t))
    #     self.assertTrue(np.array_equal(self.plot.t["feature1d"], t))
    #
    #
    #     self.assertTrue(np.allclose(self.plot.E_compare["pc"]["directComparison"], U, atol=0.05))
    #     self.assertTrue(np.allclose(self.plot.E_compare["pc"]["feature1d"], feature.feature1d(), atol=0.001))
    #
    #     self.assertTrue(np.allclose(self.plot.Var_compare["pc"]["directComparison"], np.zeros(10) + 0.1, atol=0.01))
    #     self.assertTrue(np.allclose(self.plot.Var_compare["pc"]["feature1d"], np.zeros(10), atol=0.001))
    #
    #
    #     self.assertTrue(np.all(np.less(self.plot.p_05["directComparison"], U)))
    #     self.assertTrue(np.allclose(self.plot.p_05_compare["pc"]["feature1d"], feature.feature1d(), atol=0.001))
    #
    #     self.assertTrue(np.all(np.greater(self.plot.p_95_compare["pc"]["directComparison"], U)))
    #     self.assertTrue(np.allclose(self.plot.p_95_compare["pc"]["feature1d"], feature.feature1d(), atol=0.001))
    #
    #     self.assertTrue(self.plot.sensitivity_compare["pc"]["directComparison"].shape, (10, 2))
    #     self.assertTrue(self.plot.sensitivity_compare["pc"]["feature1d"].shape, (10, 2))
    #
    #     self.assertEqual(len(self.plot.features_0d), 2)
    #     self.assertEqual(len(self.plot.features_1d), 2)
    #
    #     self.assertEqual(len(self.plot.uncertain_parameters), 2)
    #     self.assertTrue(self.plot.loaded_flag)
    #
    #
    #     self.assertEqual(self.plot.t_compare.keys(), ["pc", 'mc_10', 'mc_100'])
    #     self.assertEqual(self.plot.E_compare.keys(), ["pc", 'mc_10', 'mc_100'])
    #     self.assertEqual(self.plot.Var_compare.keys(), ["pc", 'mc_10', 'mc_100'])
    #     self.assertEqual(self.plot.p_05_compare.keys(), ["pc", 'mc_10', 'mc_100'])
    #     self.assertEqual(self.plot.p_95_compare.keys(), ["pc", 'mc_10', 'mc_100'])
    #     self.assertEqual(self.plot.sensitivity_compare.keys(), ["pc", 'mc_10', 'mc_100'])
    #
    #
    # def test_plotCompareMeanDirectComparison(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareMean(feature="directComparison")
    #
    #     self.compare_plot("directComparison_mean_compare", compare=True)
    #
    #
    # def test_plotCompareMeanFeature1d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareMean(feature="feature1d")
    #
    #     self.compare_plot("feature1d_mean_compare", compare=True)
    #
    #
    # def test_plotCompareVarianceDirectComparison(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareVariance(feature="directComparison")
    #
    #     self.compare_plot("directComparison_variance_compare", compare=True)
    #
    #
    # def test_plotCompareVarianceFeature1d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareVariance(feature="feature1d")
    #
    #     self.compare_plot("feature1d_variance_compare", compare=True)
    #
    #
    #
    # def test_plotCompareMeanAndVarianceDirectComparison(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareMeanAndVariance(feature="directComparison")
    #
    #     self.compare_plot("directComparison_mean-variance_compare", compare=True)
    #
    #
    # def test_plotCompareMeanAndVarianceFeature1d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareMeanAndVariance(feature="feature1d")
    #
    #     self.compare_plot("feature1d_mean-variance_compare", compare=True)
    #
    #
    #
    # def test_plotCompareConfidenceIntervalDirectComparison(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareConfidenceInterval(feature="directComparison")
    #
    #     self.compare_plot("directComparison_confidence-interval_compare", compare=True)
    #
    #
    #
    # def test_plotCompareConfidenceIntervalFeature1d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareConfidenceInterval(feature="feature1d")
    #
    #     self.compare_plot("feature1d_confidence-interval_compare", compare=True)
    #
    #
    #
    #
    # # # TODO not implemented sensitivity for MC
    # # def test_plotCompareSensitivity(self):
    # #     self.plot.loadCompareData("TestingModel1d",
    # #                               compare_folders=["pc", "mc_10", "mc_100"])
    # #
    # #     self.plot.plotCompareSensitivity(feature="directComparison", show=True)
    #
    #
    #
    #
    #
    # def test_CompareAttributeFeature0dMean(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareAttributeFeature0d(feature="feature0d", attribute="E",
    #                                             attribute_name="mean")
    #
    #     self.compare_plot("feature0d_mean_compare", compare=True)
    #
    #
    # def test_CompareAttributeFeature0dvariance(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareAttributeFeature0d(feature="feature0d", attribute="Var",
    #                                             attribute_name="variance")
    #
    #     self.compare_plot("feature0d_variance_compare", compare=True)
    #
    #
    #
    # def test_CompareAttributeFeature0dError(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     with self.assertRaises(ValueError):
    #         self.plot.plotCompareAttributeFeature0d(feature="feature0d", attribute="not_existing",
    #                                                 attribute_name="not existing")
    #
    #
    # def test_CompareMeanFeature0d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareMeanFeature0d(feature="feature0d")
    #
    #     self.compare_plot("feature0d_mean_compare", compare=True)
    #
    #
    # def test_CompareVarianceFeature0d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareVarianceFeature0d(feature="feature0d")
    #
    #     self.compare_plot("feature0d_variance_compare", compare=True)
    #
    #
    #
    #
    # def test_plotCompareConfidenceIntervalFeature0d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareConfidenceIntervalFeature0d(feature="feature0d")
    #
    #     self.compare_plot("feature0d_confidence-interval_compare", compare=True)
    #
    #
    #
    #
    #
    # def test_plotCompareFractionalFeature1dMean(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareAttributeFeature1dFractional(feature="directComparison",
    #                                                       attribute="E", attribute_name="mean")
    #
    #     self.compare_plot("directComparison_mean_compare_fractional", compare=True)
    #
    #
    # def test_plotCompareFractionalFeature1dVariance(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareAttributeFeature1dFractional(feature="directComparison",
    #                                                       attribute="Var", attribute_name="variance")
    #
    #     self.compare_plot("directComparison_variance_compare_fractional", compare=True)
    #
    #
    # def test_plotCompareFractionalMean(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalMean(feature="directComparison")
    #
    #
    #     self.compare_plot("directComparison_mean_compare_fractional", compare=True)
    #
    #
    #
    # def test_plotCompareFractionalVariance(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalVariance(feature="directComparison")
    #     self.compare_plot("directComparison_variance_compare_fractional", compare=True)
    #
    #
    #
    #
    # def test_plotCompareFractionalConfidenceInterval(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalConfidenceInterval("directComparison")
    #     self.compare_plot("directComparison_confidence-interval_compare_fractional",
    #                       compare=True)
    #
    #
    # def test_fractional_difference(self):
    #
    #     value = self.plot._fractional_difference(2., 1)
    #     self.assertEqual(value, 0.5)
    #
    #
    # def test_plotCompareFractionalAttributeFeature0dMean(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalAttributeFeature0d(feature="feature0d",
    #                                                       attribute="E",
    #                                                       attribute_name="mean")
    #     self.compare_plot("feature0d_mean_compare_fractional", compare=True)
    #
    #
    # def test_plotCompareFractionalAttributeFeature0dVariance(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalAttributeFeature0d(feature="feature0d",
    #                                                       attribute="Var",
    #                                                       attribute_name="variance")
    #     self.compare_plot("feature0d_variance_compare_fractional", compare=True)
    #
    #
    #
    # def test_plotCompareFractionalMeanFeature0d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalMeanFeature0d(feature="feature0d")
    #     self.compare_plot("feature0d_mean_compare_fractional", compare=True)
    #
    #
    #
    # def test_plotCompareFractionalVarianceFeature0d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalVarianceFeature0d(feature="feature0d")
    #     self.compare_plot("feature0d_variance_compare_fractional", compare=True)
    #
    #
    # def test_plotCompareFractionalConfidenceIntervalFeature0d(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractionalConfidenceIntervalFeature0d(feature="feature0d")
    #     self.compare_plot("feature0d_confidence-interval_compare_fractional",
    #                       compare=True)
    #
    #
    #
    #
    # def test_Compare1dFeatures(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompare1dFeatures()
    #
    #     self.compare_plot("directComparison_mean_compare", compare=True)
    #     self.compare_plot("directComparison_variance_compare", compare=True)
    #     self.compare_plot("directComparison_mean-variance_compare", compare=True)
    #     self.compare_plot("directComparison_confidence-interval_compare", compare=True)
    #
    #     self.compare_plot("feature1d_mean_compare", compare=True)
    #     self.compare_plot("feature1d_variance_compare", compare=True)
    #     self.compare_plot("feature1d_mean-variance_compare", compare=True)
    #     self.compare_plot("feature1d_confidence-interval_compare", compare=True)
    #
    #
    # def test_plotCompareFractional1dFeatures(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractional1dFeatures()
    #
    #     self.compare_plot("directComparison_mean_compare_fractional", compare=True)
    #     self.compare_plot("directComparison_variance_compare_fractional", compare=True)
    #     self.compare_plot("directComparison_confidence-interval_compare_fractional", compare=True)
    #
    #     self.compare_plot("feature1d_mean_compare_fractional", compare=True)
    #     self.compare_plot("feature1d_variance_compare_fractional", compare=True)
    #     self.compare_plot("feature1d_confidence-interval_compare_fractional", compare=True)
    #
    #
    # def test_Compare0dFeatures(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompare0dFeatures()
    #
    #     self.compare_plot("feature0d_mean_compare", compare=True)
    #     self.compare_plot("feature0d_variance_compare", compare=True)
    #     self.compare_plot("feature0d_confidence-interval_compare", compare=True)
    #
    #
    # def test_plotCompareFractional0dFeatures(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractional0dFeatures()
    #
    #     self.compare_plot("feature0d_mean_compare_fractional", compare=True)
    #     self.compare_plot("feature0d_variance_compare_fractional", compare=True)
    #     self.compare_plot("feature0d_confidence-interval_compare_fractional", compare=True)
    #
    #
    # def test_plotCompareFractional(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareFractional()
    #
    #     self.compare_plot("feature0d_mean_compare_fractional", compare=True)
    #     self.compare_plot("feature0d_variance_compare_fractional", compare=True)
    #     self.compare_plot("feature0d_confidence-interval_compare_fractional", compare=True)
    #
    #     self.compare_plot("directComparison_mean_compare_fractional", compare=True)
    #     self.compare_plot("directComparison_variance_compare_fractional", compare=True)
    #     self.compare_plot("directComparison_confidence-interval_compare_fractional", compare=True)
    #
    #     self.compare_plot("feature1d_mean_compare_fractional", compare=True)
    #     self.compare_plot("feature1d_variance_compare_fractional", compare=True)
    #     self.compare_plot("feature1d_confidence-interval_compare_fractional", compare=True)
    #
    #
    # def test_plotCompare(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompare()
    #
    #     self.compare_plot("feature0d_mean_compare", compare=True)
    #     self.compare_plot("feature0d_variance_compare", compare=True)
    #     self.compare_plot("feature0d_confidence-interval_compare", compare=True)
    #
    #     self.compare_plot("directComparison_mean_compare", compare=True)
    #     self.compare_plot("directComparison_variance_compare", compare=True)
    #     self.compare_plot("directComparison_mean-variance_compare", compare=True)
    #     self.compare_plot("directComparison_confidence-interval_compare", compare=True)
    #
    #     self.compare_plot("feature1d_mean_compare", compare=True)
    #     self.compare_plot("feature1d_variance_compare", compare=True)
    #     self.compare_plot("feature1d_mean-variance_compare", compare=True)
    #     self.compare_plot("feature1d_confidence-interval_compare", compare=True)
    #
    #
    #
    # def test_plotCompareAll(self):
    #     # self.plot.loadCompareData("TestingModel1d",
    #     #                           compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareAll("TestingModel1d",
    #                              compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.compare_plot("feature0d_mean_compare", compare=True)
    #     self.compare_plot("feature0d_variance_compare", compare=True)
    #     self.compare_plot("feature0d_confidence-interval_compare", compare=True)
    #
    #     self.compare_plot("directComparison_mean_compare", compare=True)
    #     self.compare_plot("directComparison_variance_compare", compare=True)
    #     self.compare_plot("directComparison_mean-variance_compare", compare=True)
    #     self.compare_plot("directComparison_confidence-interval_compare", compare=True)
    #
    #     self.compare_plot("feature1d_mean_compare", compare=True)
    #     self.compare_plot("feature1d_variance_compare", compare=True)
    #     self.compare_plot("feature1d_mean-variance_compare", compare=True)
    #     self.compare_plot("feature1d_confidence-interval_compare", compare=True)
    #
    #
    #     self.compare_plot("feature0d_mean_compare_fractional", compare=True)
    #     self.compare_plot("feature0d_variance_compare_fractional", compare=True)
    #     self.compare_plot("feature0d_confidence-interval_compare_fractional", compare=True)
    #
    #     self.compare_plot("directComparison_mean_compare_fractional", compare=True)
    #     self.compare_plot("directComparison_variance_compare_fractional", compare=True)
    #     self.compare_plot("directComparison_confidence-interval_compare_fractional", compare=True)
    #
    #     self.compare_plot("feature1d_mean_compare_fractional", compare=True)
    #     self.compare_plot("feature1d_variance_compare_fractional", compare=True)
    #     self.compare_plot("feature1d_confidence-interval_compare_fractional", compare=True)




# TODO test for creating gif
# TODO test combined features 0 for many features

# TODO test plotAllDataFromExploration
# TODO test plotAllDataInFolder



if __name__ == "__main__":
    unittest.main()
