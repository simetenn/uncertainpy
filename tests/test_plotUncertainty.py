import numpy as np
import glob
import os
import unittest
import subprocess
import shutil


from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from TestingFeatures import TestingFeatures
from TestingModel import TestingModel1d
from uncertainpy import Data

class TestPlotUncertainpy(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.test_data_dir = os.path.join(self.folder, "data")
        self.data_file = "TestingModel1d"
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.plot = PlotUncertainty(data_dir=self.test_data_dir,
                                    output_dir_figures=self.output_test_dir,
                                    verbose_level="error")



    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)



    def compare_plot(self, name):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures/TestingModel1d",
                                    name + ".png")

        plot_file = os.path.join(self.output_test_dir, self.data_file,
                                 name + ".png")

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)



    def test_plotTotalSensitivityGrid1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotTotalSensitivityGrid(hardcopy=True, sensitivity="sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")

    def test_plotTotalSensitivityGridT(self):
        self.plot.loadData(self.data_file)

        self.plot.plotTotalSensitivityGrid(hardcopy=True, sensitivity="sensitivity_t")

        self.compare_plot("total-sensitivity_t_grid")


    def test_plotTotalSensitivity1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotTotalSensitivity(feature="feature1d",
                                       sensitivity="sensitivity_1",
                                       hardcopy=True)

        self.compare_plot("feature1d_total-sensitivity_1")

    def test_plotTotalSensitivityT(self):
        self.plot.loadData(self.data_file)

        self.plot.plotTotalSensitivity(feature="feature1d",
                                       sensitivity="sensitivity_t",
                                       hardcopy=True)

        self.compare_plot("feature1d_total-sensitivity_t")


    def test_plotTotalSensitivityAllFeatures1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotTotalSensitivityAllFeatures(hardcopy=True, sensitivity="sensitivity_1")

        self.compare_plot("directComparison_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")
        self.compare_plot("featureInvalid_total-sensitivity_1")


    def test_plotTotalSensitivityAllFeaturesT(self):
        self.plot.loadData(self.data_file)

        self.plot.plotTotalSensitivityAllFeatures(hardcopy=True, sensitivity="sensitivity_t")

        self.compare_plot("directComparison_total-sensitivity_t")
        self.compare_plot("feature0d_total-sensitivity_t")
        self.compare_plot("feature1d_total-sensitivity_t")
        self.compare_plot("feature2d_total-sensitivity_t")
        self.compare_plot("featureInvalid_total-sensitivity_t")


    def test_plotSimulatorResults(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.plot.data.t["directComparison"] = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.plot.data.U["directComparison"] = [U, U, U, U, U]

        self.plot.plotSimulatorResults()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/U.png")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            result = subprocess.call(["diff", plot, compare_file])

            self.assertEqual(result, 0)

            plot_count += 1

        self.assertEqual(plot_count, 5)



    def test_init(self):
        plot = PlotUncertainty(data_dir=self.test_data_dir,
                               output_dir_figures=self.output_test_dir,
                               verbose_level="error")

        self.assertIsInstance(plot, PlotUncertainty)


    def test_loadData(self):
        self.plot.loadData("TestingModel1d")

        self.assertData()


    def test_setData(self):
        data = Data()

        data.load(os.path.join(self.test_data_dir, self.data_file + ".h5"))

        self.plot.setData(data)

        self.assertData()


    def assertData(self):
        model = TestingModel1d()
        model.run()
        t = model.t
        U = model.U

        feature = TestingFeatures()

        # TODO currently only tests for directComparison and feature1d,
        # does not test data of the rest
        # TODO test total_sensitivity

        self.assertTrue(np.array_equal(self.plot.data.t["directComparison"], t))
        self.assertTrue(np.array_equal(self.plot.data.t["feature1d"], t))


        self.assertTrue(self.plot.data.U["directComparison"].shape, (10, 21))
        self.assertTrue(self.plot.data.U["feature1d"].shape, (10, 21))

        self.assertTrue(np.allclose(self.plot.data.E["directComparison"], U, atol=0.001))
        self.assertTrue(np.allclose(self.plot.data.E["feature1d"], feature.feature1d(), atol=0.001))

        self.assertTrue(np.allclose(self.plot.data.Var["directComparison"], np.zeros(10) + 0.1, atol=0.01))
        self.assertTrue(np.allclose(self.plot.data.Var["feature1d"], np.zeros(10), atol=0.001))


        self.assertTrue(np.all(np.less(self.plot.data.p_05["directComparison"], U)))
        self.assertTrue(np.allclose(self.plot.data.p_05["feature1d"], feature.feature1d(), atol=0.001))

        self.assertTrue(np.all(np.greater(self.plot.data.p_95["directComparison"], U)))
        self.assertTrue(np.allclose(self.plot.data.p_95["feature1d"], feature.feature1d(), atol=0.001))

        self.assertTrue(self.plot.data.sensitivity_1["directComparison"].shape, (10, 2))
        self.assertTrue(self.plot.data.sensitivity_1["feature1d"].shape, (10, 2))

        self.assertTrue(self.plot.data.sensitivity_t["directComparison"].shape, (10, 2))
        self.assertTrue(self.plot.data.sensitivity_t["feature1d"].shape, (10, 2))


        self.assertEqual(len(self.plot.data.features_0d), 2)
        self.assertEqual(len(self.plot.data.features_1d), 2)

        self.assertEqual(len(self.plot.data.uncertain_parameters), 2)
        self.assertTrue(self.plot.loaded_flag)



    def test_plotAttributeFeature1dError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotAttributeFeature1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotAttributeFeature1d(feature="feature2d")

        with self.assertRaises(ValueError):
            self.plot.plotAttributeFeature1d(attribute="test")


    def test_plotAttributeFeature1dMean(self):
        self.plot.loadData(self.data_file)

        self.plot.plotAttributeFeature1d(feature="directComparison",
                                         attribute="E",
                                         attribute_name="mean")

        self.compare_plot("directComparison_mean")


    def test_plotAttributeFeature1dVariance(self):
        self.plot.loadData(self.data_file)

        self.plot.plotAttributeFeature1d(feature="directComparison",
                                         attribute="Var",
                                         attribute_name="variance")

        self.compare_plot("directComparison_variance")



    def test_plotMeanError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotMean(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotMean(feature="feature2d")


    def test_plotMeanDirectComparison(self):
        self.plot.loadData(self.data_file)

        self.plot.plotMean(feature="directComparison")

        self.compare_plot("directComparison_mean")


    def test_plotMeanfeature1d(self):
        self.plot.loadData(self.data_file)

        self.plot.plotMean(feature="feature1d")

        self.compare_plot("feature1d_mean")


    def test_plotVarianceError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotVariance(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotVariance(feature="feature2d")


    def test_plotVarianceDirectComparison(self):
        self.plot.loadData(self.data_file)

        self.plot.plotVariance(feature="directComparison")

        self.compare_plot("directComparison_variance")


    def test_plotVariancefeature1d(self):
        self.plot.loadData(self.data_file)

        self.plot.plotVariance(feature="feature1d")

        self.compare_plot("feature1d_variance")



    def test_plotMeanAndVarianceError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotMeanAndVariance(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotMeanAndVariance(feature="feature2d")


    def test_plotMeanAndVarianceDirectComparison(self):
        self.plot.loadData(self.data_file)

        self.plot.plotMeanAndVariance(feature="directComparison")

        self.compare_plot("directComparison_mean-variance")


    def test_plotMeanAndVariancefeature1d(self):
        self.plot.loadData(self.data_file)

        self.plot.plotMeanAndVariance(feature="feature1d")

        self.compare_plot("feature1d_mean-variance")



    def test_plotConfidenceIntervalDirectComparison(self):
        self.plot.loadData(self.data_file)

        self.plot.plotConfidenceInterval(feature="directComparison")

        self.compare_plot("directComparison_confidence-interval")


    def test_plotConfidenceIntervalFeature0d(self):
        self.plot.loadData(self.data_file)

        self.plot.plotConfidenceInterval(feature="feature1d")

        self.compare_plot("feature1d_confidence-interval")


    def test_plotConfidenceIntervalError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotConfidenceInterval(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotConfidenceInterval(feature="feature2d")


    def test_plotSensitivityDirectComparison1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivity(feature="directComparison", sensitivity="sensitivity_1")

        self.compare_plot("directComparison_sensitivity_1_a")
        self.compare_plot("directComparison_sensitivity_1_b")


    def test_plotSensitivityDirectComparisonT(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivity(feature="directComparison", sensitivity="sensitivity_t")

        self.compare_plot("directComparison_sensitivity_t_a")
        self.compare_plot("directComparison_sensitivity_t_b")


    def test_plotSensitivityFeature1d1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivity(feature="feature1d", sensitivity="sensitivity_1")


        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")


    def test_plotSensitivityFeature1d1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivity(feature="feature1d", sensitivity="sensitivity_t")


        self.compare_plot("feature1d_sensitivity_t_a")
        self.compare_plot("feature1d_sensitivity_t_b")


    def test_plotSensitivityError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotSensitivity(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotSensitivity(feature="feature2d")



    def test_plotSensitivityCombinedDirectComparison1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivityCombined(feature="directComparison", sensitivity="sensitivity_1")

        self.compare_plot("directComparison_sensitivity_1")


    def test_plotSensitivityCombinedDirectComparisonT(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivityCombined(feature="directComparison", sensitivity="sensitivity_t")

        self.compare_plot("directComparison_sensitivity_t")


    def test_plotSensitivityCombinedFeature1d1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivityCombined(feature="feature1d", sensitivity="sensitivity_1")

        self.compare_plot("feature1d_sensitivity_1")

    def test_plotSensitivityCombinedFeature1dT(self):
        self.plot.loadData(self.data_file)

        self.plot.plotSensitivityCombined(feature="feature1d", sensitivity="sensitivity_t")

        self.compare_plot("feature1d_sensitivity_t")


    def test_plotSensitivityCombinedError(self):
        self.plot.loadData(self.data_file)

        with self.assertRaises(ValueError):
            self.plot.plotSensitivityCombined(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotSensitivityCombined(feature="feature2d")



    def test_plot1dFeatures(self):
        self.plot.loadData(self.data_file)

        self.plot.plot1dFeatures(sensitivity="sensitivity_1")

        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")
        self.compare_plot("feature1d_sensitivity_1")



    def test_plot0dFeature(self):
        self.plot.loadData(self.data_file)

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_1")

        self.compare_plot("feature0d_sensitivity_1")

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_t")

        self.compare_plot("feature0d_sensitivity_t")


        with self.assertRaises(ValueError):
            self.plot.plot0dFeature(feature="feature1d")


    def test_plot0dFeatures1(self):
        self.plot.loadData(self.data_file)

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_1")

        self.compare_plot("feature0d_sensitivity_1")



    def test_plot0dFeaturesT(self):
        self.plot.loadData(self.data_file)

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_t")

        self.compare_plot("feature0d_sensitivity_t")


    def test_plotResults(self):
        self.plot.loadData(self.data_file)

        self.plot.plotResults(sensitivity="sensitivity_1")

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("featureInvalid_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_plotAllData1(self):
        self.plot.loadData(self.data_file)

        self.plot.plotAllData("sensitivity_1")

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

        self.compare_plot("featureInvalid_total-sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")

    def test_plotAllDataT(self):
        self.plot.loadData(self.data_file)

        self.plot.plotAllData("sensitivity_t")


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

        self.compare_plot("featureInvalid_total-sensitivity_t")

        self.compare_plot("total-sensitivity_t_grid")


    def test_plotAllDataSensitivity(self):
        self.plot.loadData(self.data_file)

        self.plot.plotAllDataSensitivity()

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
        self.compare_plot("featureInvalid_total-sensitivity_1")

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
        self.compare_plot("featureInvalid_total-sensitivity_t")

        self.compare_plot("total-sensitivity_t_grid")


# TODO test combined features 0 for many features

# TODO test plotAllDataFromExploration
# TODO test plotAllDataInFolder



if __name__ == "__main__":
    unittest.main()
