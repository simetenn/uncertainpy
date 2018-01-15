import numpy as np
import os
import unittest
import subprocess
import shutil
import scipy

import matplotlib
matplotlib.use('Agg')

from uncertainpy.plotting.plotUncertaintyCompare import PlotUncertaintyCompare

from .testing_classes import TestingFeatures
from .testing_classes import TestingModel1d

class TestPlotUncertainpyCompare(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.test_data_dir = os.path.join(self.folder, "data")
        self.data_file = "test_plot_data"
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.plot = PlotUncertaintyCompare(data_dir=self.test_data_dir,
                                           figure_folder=self.output_test_dir,
                                           verbose_level="error")



    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)



    def compare_plot(self, name):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures/compare",
                                    name + ".png")
        plot_file = os.path.join(self.output_test_dir,
                                 "compare", name + ".png")

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)





    def test_adaptiveFeatures(self):
        self.plot.E_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                      "constant_feature": np.zeros(10)},
                               "mc_10": {"adaptive_feature": np.zeros(100),
                                         "constant_feature": np.zeros(10)},
                               "mc_100": {"adaptive_feature": np.zeros(10),
                                          "constant_feature": np.zeros(10)}}

        self.plot.compare_folders = ["pc", "mc_10", "mc_100"]
        self.plot.data.features_1d = ["adaptive_feature", "constant_feature"]


        self.assertEqual(self.plot.adaptiveFeatures(), ["adaptive_feature"])

    def test_setData(self):
        self.plot.E_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                      "constant_feature": np.zeros(10)},
                               "mc_10": {"adaptive_feature": np.zeros(100),
                                         "constant_feature": np.zeros(10)},
                               "mc_100": {"adaptive_feature": np.zeros(10),
                                          "constant_feature": np.zeros(10)}}

        data = {"pc": np.arange(100), "mc_10": np.arange(100), "mc_100": np.arange(100)}

        self.plot.setData(self.plot.E_compare, data, "adaptive_feature")


        self.assertTrue(np.array_equal(self.plot.E_compare["pc"]["adaptive_feature"],
                                       np.arange(100)))
        self.assertTrue(np.array_equal(self.plot.E_compare["mc_10"]["adaptive_feature"],
                                       np.arange(100)))
        self.assertTrue(np.array_equal(self.plot.E_compare["mc_100"]["adaptive_feature"],
                                       np.arange(100)))

        self.assertTrue(np.array_equal(self.plot.E_compare["pc"]["constant_feature"], np.zeros(10)))
        self.assertTrue(np.array_equal(self.plot.E_compare["mc_10"]["constant_feature"], np.zeros(10)))
        self.assertTrue(np.array_equal(self.plot.E_compare["mc_100"]["constant_feature"], np.zeros(10)))


    def test_getData(self):
        self.plot.E_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                      "constant_feature": np.zeros(10)},
                               "mc_10": {"adaptive_feature": np.zeros(100),
                                         "constant_feature": np.zeros(10)},
                               "mc_100": {"adaptive_feature": np.zeros(10),
                                          "constant_feature": np.zeros(10)}}


        self.plot.compare_folders = ["pc", "mc_10", "mc_100"]

        constant_result = self.plot.getData(self.plot.E_compare, "constant_feature")
        self.assertTrue(np.array_equal(constant_result["pc"], np.zeros(10)))
        self.assertTrue(np.array_equal(constant_result["mc_10"], np.zeros(10)))
        self.assertTrue(np.array_equal(constant_result["mc_100"], np.zeros(10)))

        adaptive_result = self.plot.getData(self.plot.E_compare, "adaptive_feature")
        self.assertTrue(np.array_equal(adaptive_result["pc"], np.zeros(10)))
        self.assertTrue(np.array_equal(adaptive_result["mc_10"], np.zeros(100)))
        self.assertTrue(np.array_equal(adaptive_result["mc_100"], np.zeros(10)))


    def test_creatInterpolation(self):
        self.plot.compare_folders = ["pc", "mc_10", "mc_100"]

        data = {"pc": np.zeros(10), "mc_10": np.zeros(100), "mc_100": np.zeros(10)}
        t = {"pc": np.arange(10), "mc_10": np.arange(100), "mc_100": np.arange(10)}


        result = self.plot.createInterpolation(data, t)

        self.assertIsInstance(result["pc"], scipy.interpolate.fitpack2.InterpolatedUnivariateSpline)
        self.assertIsInstance(result["mc_10"], scipy.interpolate.fitpack2.InterpolatedUnivariateSpline)
        self.assertIsInstance(result["mc_100"], scipy.interpolate.fitpack2.InterpolatedUnivariateSpline)



    def test_perform_interpolation(self):
        self.plot.compare_folders = ["pc", "mc_10", "mc_100"]

        data = {"pc": np.zeros(10), "mc_10": np.zeros(100), "mc_100": np.zeros(10)}
        t = {"pc": np.arange(10), "mc_10": np.arange(100), "mc_100": np.arange(10)}

        interpolations = self.plot.createInterpolation(data, t)

        t, results = self.plot.perform_interpolation(t, interpolations)

        self.assertTrue(np.array_equal(t["pc"], np.arange(100)))
        self.assertTrue(np.array_equal(t["mc_10"], np.arange(100)))
        self.assertTrue(np.array_equal(t["mc_100"], np.arange(100)))
        self.assertTrue(np.array_equal(results["pc"], np.zeros(100)))
        self.assertTrue(np.array_equal(results["mc_10"], np.zeros(100)))
        self.assertTrue(np.array_equal(results["mc_100"], np.zeros(100)))



    def test_interpolateData(self):
        self.plot.E_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                      "constant_feature": np.zeros(10)},
                               "mc_10": {"adaptive_feature": np.zeros(100),
                                         "constant_feature": np.zeros(10)},
                               "mc_100": {"adaptive_feature": np.zeros(10),
                                          "constant_feature": np.zeros(10)}}

        self.plot.t_compare = {"pc": {"adaptive_feature": np.arange(10),
                                      "constant_feature": np.arange(10)},
                               "mc_10": {"adaptive_feature": np.arange(100),
                                         "constant_feature": np.arange(10)},
                               "mc_100": {"adaptive_feature": np.arange(10),
                                          "constant_feature": np.arange(10)}}


        self.plot.loaded_compare_flag = True
        self.plot.compare_folders = ["pc", "mc_10", "mc_100"]

        self.plot.interpolateData(self.plot.E_compare, "adaptive_feature")

        self.compare_data(self.plot.E_compare)


    def compare_data(self, data):
        self.assertTrue(np.array_equal(data["pc"]["adaptive_feature"],
                                       np.zeros(100)))
        self.assertTrue(np.array_equal(data["mc_10"]["adaptive_feature"],
                                       np.zeros(100)))
        self.assertTrue(np.array_equal(data["mc_100"]["adaptive_feature"],
                                       np.zeros(100)))

        self.assertTrue(np.array_equal(data["pc"]["constant_feature"], np.zeros(10)))
        self.assertTrue(np.array_equal(data["mc_10"]["constant_feature"], np.zeros(10)))
        self.assertTrue(np.array_equal(data["mc_100"]["constant_feature"], np.zeros(10)))



    def test_interpolateAllData(self):
        self.plot.E_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                      "constant_feature": np.zeros(10)},
                               "mc_10": {"adaptive_feature": np.zeros(100),
                                         "constant_feature": np.zeros(10)},
                               "mc_100": {"adaptive_feature": np.zeros(10),
                                          "constant_feature": np.zeros(10)}}


        self.plot.Var_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                        "constant_feature": np.zeros(10)},
                                 "mc_10": {"adaptive_feature": np.zeros(100),
                                           "constant_feature": np.zeros(10)},
                                 "mc_100": {"adaptive_feature": np.zeros(10),
                                            "constant_feature": np.zeros(10)}}

        self.plot.percentile_5_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                         "constant_feature": np.zeros(10)},
                                  "mc_10": {"adaptive_feature": np.zeros(100),
                                            "constant_feature": np.zeros(10)},
                                  "mc_100": {"adaptive_feature": np.zeros(10),
                                             "constant_feature": np.zeros(10)}}

        self.plot.percentile_95_compare = {"pc": {"adaptive_feature": np.zeros(10),
                                         "constant_feature": np.zeros(10)},
                                  "mc_10": {"adaptive_feature": np.zeros(100),
                                            "constant_feature": np.zeros(10)},
                                  "mc_100": {"adaptive_feature": np.zeros(10),
                                             "constant_feature": np.zeros(10)}}


        self.plot.sensitivity_compare = {"pc": {"adaptive_feature": None,
                                                "constant_feature": None},
                                         "mc_10": {"adaptive_feature": None,
                                                   "constant_feature": None},
                                         "mc_100": {"adaptive_feature": None,
                                                    "constant_feature": None}}

        self.plot.t_compare = {"pc": {"adaptive_feature": np.arange(10),
                                      "constant_feature": np.arange(10)},
                               "mc_10": {"adaptive_feature": np.arange(100),
                                         "constant_feature": np.arange(10)},
                               "mc_100": {"adaptive_feature": np.arange(10),
                                          "constant_feature": np.arange(10)}}


        self.plot.loaded_compare_flag = True
        self.plot.compare_folders = ["pc", "mc_10", "mc_100"]
        self.plot.data.features_1d = ["adaptive_feature", "constant_feature"]



        self.plot.interpolateAllData()

        self.compare_data(self.plot.E_compare)
        self.compare_data(self.plot.Var_compare)
        self.compare_data(self.plot.percentile_5_compare)
        self.compare_data(self.plot.percentile_95_compare)


        self.assertEqual(self.plot.sensitivity_compare, {"pc": {"adaptive_feature": None,
                                                                "constant_feature": None},
                                                         "mc_10": {"adaptive_feature": None,
                                                                   "constant_feature": None},
                                                         "mc_100": {"adaptive_feature": None,
                                                                    "constant_feature": None}})






    def test_loadCompareData(self):
        self.plot.loadCompareData("TestingModel1d", ["pc", "mc_10", "mc_100"])

        model = TestingModel1d()
        model.run()
        t = model.t
        values = model.U

        feature = TestingFeatures()

        self.assertTrue(np.array_equal(self.plot.data.t["directComparison"], t))
        self.assertTrue(np.array_equal(self.plot.data.t["feature1d"], t))


        self.assertTrue(np.allclose(self.plot.E_compare["pc"]["directComparison"], U, atol=0.05))
        self.assertTrue(np.allclose(self.plot.E_compare["pc"]["feature1d"], feature.feature1d(), atol=0.001))

        self.assertTrue(np.allclose(self.plot.Var_compare["pc"]["directComparison"], np.zeros(10) + 0.1, atol=0.01))
        self.assertTrue(np.allclose(self.plot.Var_compare["pc"]["feature1d"], np.zeros(10), atol=0.001))


        self.assertTrue(np.all(np.less(self.plot.percentile_5_compare["pc"]["directComparison"], U)))
        self.assertTrue(np.allclose(self.plot.percentile_5_compare["pc"]["feature1d"], feature.feature1d(), atol=0.001))

        self.assertTrue(np.all(np.greater(self.plot.percentile_95_compare["pc"]["directComparison"], U)))
        self.assertTrue(np.allclose(self.plot.percentile_95_compare["pc"]["feature1d"], feature.feature1d(), atol=0.001))

        self.assertTrue(self.plot.sensitivity_compare["pc"]["directComparison"].shape, (10, 2))
        self.assertTrue(self.plot.sensitivity_compare["pc"]["feature1d"].shape, (10, 2))

        self.assertEqual(len(self.plot.data.features_0d), 2)
        self.assertEqual(len(self.plot.data.features_1d), 2)

        self.assertEqual(len(self.plot.data.uncertain_parameters), 2)
        self.assertTrue(self.plot.loaded_flag)


        self.assertEqual(self.plot.t_compare.keys(), ["pc", 'mc_10', 'mc_100'])
        self.assertEqual(self.plot.E_compare.keys(), ["pc", 'mc_10', 'mc_100'])
        self.assertEqual(self.plot.Var_compare.keys(), ["pc", 'mc_10', 'mc_100'])
        self.assertEqual(self.plot.percentile_5_compare.keys(), ["pc", 'mc_10', 'mc_100'])
        self.assertEqual(self.plot.percentile_95_compare.keys(), ["pc", 'mc_10', 'mc_100'])
        self.assertEqual(self.plot.sensitivity_compare.keys(), ["pc", 'mc_10', 'mc_100'])


    def test_plotCompareMeanDirectComparison(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareMean(feature="directComparison")


        self.compare_plot("directComparison_mean_compare")


    def test_plotCompareMeanFeature1d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareMean(feature="feature1d")

        self.compare_plot("feature1d_mean_compare")


    def test_plotCompareVarianceDirectComparison(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareVariance(feature="directComparison")

        self.compare_plot("directComparison_variance_compare")


    def test_plotCompareVarianceFeature1d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareVariance(feature="feature1d")

        self.compare_plot("feature1d_variance_compare")



    def test_plotCompareMeanAndVarianceDirectComparison(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareMeanAndVariance(feature="directComparison")

        self.compare_plot("directComparison_mean-variance_compare")


    def test_plotCompareMeanAndVarianceFeature1d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareMeanAndVariance(feature="feature1d")

        self.compare_plot("feature1d_mean-variance_compare")



    def test_plotCompareConfidenceIntervalDirectComparison(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareConfidenceInterval(feature="directComparison")

        self.compare_plot("directComparison_prediction-interval_compare")



    def test_plotCompareConfidenceIntervalFeature1d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareConfidenceInterval(feature="feature1d")

        self.compare_plot("feature1d_prediction-interval_compare")




    # # TODO not implemented sensitivity for MC
    # def test_plotCompareSensitivity(self):
    #     self.plot.loadCompareData("TestingModel1d",
    #                               compare_folders=["pc", "mc_10", "mc_100"])
    #
    #     self.plot.plotCompareSensitivity(feature="directComparison", show=True)





    def test_CompareAttributeFeature0dMean(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareAttributeFeature0d(feature="feature0d", attribute="mean",
                                                attribute_name="mean")

        self.compare_plot("feature0d_mean_compare")


    def test_CompareAttributeFeature0dvariance(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareAttributeFeature0d(feature="feature0d", attribute="Var",
                                                attribute_name="variance")

        self.compare_plot("feature0d_variance_compare")



    def test_CompareAttributeFeature0dError(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        with self.assertRaises(ValueError):
            self.plot.plotCompareAttributeFeature0d(feature="feature0d", attribute="not_existing",
                                                    attribute_name="not existing")


    def test_CompareMeanFeature0d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareMeanFeature0d(feature="feature0d")

        self.compare_plot("feature0d_mean_compare")


    def test_CompareVarianceFeature0d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareVarianceFeature0d(feature="feature0d")

        self.compare_plot("feature0d_variance_compare")




    def test_plotCompareConfidenceIntervalFeature0d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareConfidenceIntervalFeature0d(feature="feature0d")

        self.compare_plot("feature0d_prediction-interval_compare")





    def test_plotCompareFractionalFeature1dMean(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareAttributeFeature1dFractional(feature="directComparison",
                                                          attribute="mean", attribute_name="mean")

        self.compare_plot("directComparison_mean_compare_fractional")


    def test_plotCompareFractionalFeature1dVariance(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareAttributeFeature1dFractional(feature="directComparison",
                                                          attribute="Var", attribute_name="variance")

        self.compare_plot("directComparison_variance_compare_fractional")


    def test_plotCompareFractionalMean(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalMean(feature="directComparison")


        self.compare_plot("directComparison_mean_compare_fractional")



    def test_plotCompareFractionalVariance(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalVariance(feature="directComparison")
        self.compare_plot("directComparison_variance_compare_fractional")




    def test_plotCompareFractionalConfidenceInterval(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalConfidenceInterval("directComparison")
        self.compare_plot("directComparison_prediction-interval_compare_fractional")


    def test_fractional_difference(self):

        value = self.plot._fractional_difference(2., 1)
        self.assertEqual(value, 0.5)


    def test_plotCompareFractionalAttributeFeature0dMean(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalAttributeFeature0d(feature="feature0d",
                                                          attribute="mean",
                                                          attribute_name="mean")
        self.compare_plot("feature0d_mean_compare_fractional")


    def test_plotCompareFractionalAttributeFeature0dVariance(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalAttributeFeature0d(feature="feature0d",
                                                          attribute="Var",
                                                          attribute_name="variance")
        self.compare_plot("feature0d_variance_compare_fractional")



    def test_plotCompareFractionalMeanFeature0d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalMeanFeature0d(feature="feature0d")
        self.compare_plot("feature0d_mean_compare_fractional")



    def test_plotCompareFractionalVarianceFeature0d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalVarianceFeature0d(feature="feature0d")
        self.compare_plot("feature0d_variance_compare_fractional")


    def test_plotCompareFractionalConfidenceIntervalFeature0d(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractionalConfidenceIntervalFeature0d(feature="feature0d")
        self.compare_plot("feature0d_prediction-interval_compare_fractional")




    def test_Compare1dFeatures(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompare1dFeatures()

        self.compare_plot("directComparison_mean_compare")
        self.compare_plot("directComparison_variance_compare")
        self.compare_plot("directComparison_mean-variance_compare")
        self.compare_plot("directComparison_prediction-interval_compare")

        self.compare_plot("feature1d_mean_compare")
        self.compare_plot("feature1d_variance_compare")
        self.compare_plot("feature1d_mean-variance_compare")
        self.compare_plot("feature1d_prediction-interval_compare")


    def test_plotCompareFractional1dFeatures(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractional1dFeatures()

        self.compare_plot("directComparison_mean_compare_fractional")
        self.compare_plot("directComparison_variance_compare_fractional")
        self.compare_plot("directComparison_prediction-interval_compare_fractional")

        self.compare_plot("feature1d_mean_compare_fractional")
        self.compare_plot("feature1d_variance_compare_fractional")
        self.compare_plot("feature1d_prediction-interval_compare_fractional")


    def test_Compare0dFeatures(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompare0dFeatures()

        self.compare_plot("feature0d_mean_compare")
        self.compare_plot("feature0d_variance_compare")
        self.compare_plot("feature0d_prediction-interval_compare")


    def test_plotCompareFractional0dFeatures(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractional0dFeatures()

        self.compare_plot("feature0d_mean_compare_fractional")
        self.compare_plot("feature0d_variance_compare_fractional")
        self.compare_plot("feature0d_prediction-interval_compare_fractional")


    def test_plotCompareFractional(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareFractional()

        self.compare_plot("feature0d_mean_compare_fractional")
        self.compare_plot("feature0d_variance_compare_fractional")
        self.compare_plot("feature0d_prediction-interval_compare_fractional")

        self.compare_plot("directComparison_mean_compare_fractional")
        self.compare_plot("directComparison_variance_compare_fractional")
        self.compare_plot("directComparison_prediction-interval_compare_fractional")

        self.compare_plot("feature1d_mean_compare_fractional")
        self.compare_plot("feature1d_variance_compare_fractional")
        self.compare_plot("feature1d_prediction-interval_compare_fractional")


    def test_plotCompare(self):
        self.plot.loadCompareData("TestingModel1d",
                                  compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompare()

        self.compare_plot("feature0d_mean_compare")
        self.compare_plot("feature0d_variance_compare")
        self.compare_plot("feature0d_prediction-interval_compare")

        self.compare_plot("directComparison_mean_compare")
        self.compare_plot("directComparison_variance_compare")
        self.compare_plot("directComparison_mean-variance_compare")
        self.compare_plot("directComparison_prediction-interval_compare")

        self.compare_plot("feature1d_mean_compare")
        self.compare_plot("feature1d_variance_compare")
        self.compare_plot("feature1d_mean-variance_compare")
        self.compare_plot("feature1d_prediction-interval_compare")



    def test_plotCompareAll(self):
        # self.plot.loadCompareData("TestingModel1d",
        #                           compare_folders=["pc", "mc_10", "mc_100"])

        self.plot.plotCompareAll("TestingModel1d",
                                 compare_folders=["pc", "mc_10", "mc_100"])

        self.compare_plot("feature0d_mean_compare")
        self.compare_plot("feature0d_variance_compare")
        self.compare_plot("feature0d_prediction-interval_compare")

        self.compare_plot("directComparison_mean_compare")
        self.compare_plot("directComparison_variance_compare")
        self.compare_plot("directComparison_mean-variance_compare")
        self.compare_plot("directComparison_prediction-interval_compare")

        self.compare_plot("feature1d_mean_compare")
        self.compare_plot("feature1d_variance_compare")
        self.compare_plot("feature1d_mean-variance_compare")
        self.compare_plot("feature1d_prediction-interval_compare")


        self.compare_plot("feature0d_mean_compare_fractional")
        self.compare_plot("feature0d_variance_compare_fractional")
        self.compare_plot("feature0d_prediction-interval_compare_fractional")

        self.compare_plot("directComparison_mean_compare_fractional")
        self.compare_plot("directComparison_variance_compare_fractional")
        self.compare_plot("directComparison_prediction-interval_compare_fractional")

        self.compare_plot("feature1d_mean_compare_fractional")
        self.compare_plot("feature1d_variance_compare_fractional")
        self.compare_plot("feature1d_prediction-interval_compare_fractional")




if __name__ == "__main__":
    unittest.main()
