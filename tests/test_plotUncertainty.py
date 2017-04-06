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



    def test_plotTotalSensitivityGrid1(self):
        self.plot.load(self.data_file_path)

        self.plot.plotTotalSensitivityGrid(hardcopy=True, sensitivity="sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_plotTotalSensitivityGridT(self):
        self.plot.load(self.data_file_path)

        self.plot.plotTotalSensitivityGrid(hardcopy=True, sensitivity="sensitivity_t")

        self.compare_plot("total-sensitivity_t_grid")


    def test_plotTotalSensitivity1(self):
        self.plot.load(self.data_file_path)

        self.plot.plotTotalSensitivity(feature="feature1d",
                                       sensitivity="sensitivity_1",
                                       hardcopy=True)

        self.compare_plot("feature1d_total-sensitivity_1")

    def test_plotTotalSensitivityT(self):
        self.plot.load(self.data_file_path)

        self.plot.plotTotalSensitivity(feature="feature1d",
                                       sensitivity="sensitivity_t",
                                       hardcopy=True)

        self.compare_plot("feature1d_total-sensitivity_t")


    def test_plotTotalSensitivityAllFeatures1(self):
        self.plot.load(self.data_file_path)

        self.plot.plotTotalSensitivityAllFeatures(hardcopy=True, sensitivity="sensitivity_1")

        self.compare_plot("directComparison_total-sensitivity_1")
        self.compare_plot("feature0d_total-sensitivity_1")
        self.compare_plot("feature1d_total-sensitivity_1")
        self.compare_plot("feature2d_total-sensitivity_1")


    def test_plotTotalSensitivityAllFeaturesT(self):
        self.plot.load(self.data_file_path)

        self.plot.plotTotalSensitivityAllFeatures(hardcopy=True, sensitivity="sensitivity_t")

        self.compare_plot("directComparison_total-sensitivity_t")
        self.compare_plot("feature0d_total-sensitivity_t")
        self.compare_plot("feature1d_total-sensitivity_t")
        self.compare_plot("feature2d_total-sensitivity_t")


    def test_plotSimulatorResults1D(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.plot.data = Data()

        self.plot.data.t["directComparison"] = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.plot.data.U["directComparison"] = [U, U, U, U, U]
        self.plot.data.features_1d = ["directComparison"]
        self.plot.plotSimulatorResults1D()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures/U.png")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            result = subprocess.call(["diff", plot, compare_file])

            self.assertEqual(result, 0)

            plot_count += 1

        self.assertEqual(plot_count, 5)



    def test_plotSimulatorResults0DError(self):
        self.plot.data = Data()

        with self.assertRaises(ValueError):
            self.plot.plotSimulatorResults1D()

    def test_plotSimulatorResults1DError(self):
        self.plot.data = Data()

        with self.assertRaises(ValueError):
            self.plot.plotSimulatorResults1D()

    def test_plotSimulatorResultsError(self):
        self.plot.data = Data()

        with self.assertRaises(NotImplementedError):
            self.plot.plotSimulatorResults()


    def test_plotSimulatorResults_1d_model(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.plot.data = Data()

        self.plot.data.t["directComparison"] = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.plot.data.U["directComparison"] = [U, U, U, U, U]
        self.plot.data.features_1d = ["directComparison"]
        self.plot.plotSimulatorResults()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures/U.png")

        plot_count = 0
        for plot in glob.glob(os.path.join(self.output_test_dir, "simulator_results/*.png")):
            result = subprocess.call(["diff", plot, compare_file])

            self.assertEqual(result, 0)

            plot_count += 1

        self.assertEqual(plot_count, 5)


    def test_plotSimulatorResults_2d(self):
        self.plot.data = Data()

        with self.assertRaises(NotImplementedError):
            self.plot.plotSimulatorResults()


    def test_plotSimulatorResults_0d_model(self):
        self.plot.load(os.path.join(self.test_data_dir, "TestingModel0d.h5"))

        self.plot.plotSimulatorResults()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures", "simulator_results", "U" + self.figureformat)

        plot_file = os.path.join(self.output_test_dir, "simulator_results", "U" + self.figureformat)

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)


    def test_plotSimulatorResults0D(self):
        self.plot.load(os.path.join(self.test_data_dir, "TestingModel0d.h5"))

        self.plot.plotSimulatorResults()

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

        self.assertData()


    def test_set_data(self):
        data = Data()

        data.load(os.path.join(self.test_data_dir, "test_save_mock"))

        self.plot.data = data

        self.assertData()

    def test_set_output_dir(self):
        test_dir = os.path.join(self.output_test_dir, "testing")
        self.plot.output_dir = test_dir

        self.assertTrue(os.path.isdir)


    def assertData(self):
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



    def test_plotAttributeFeature1dError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.plotAttributeFeature1d(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotAttributeFeature1d(feature="feature2d")

        with self.assertRaises(ValueError):
            self.plot.plotAttributeFeature1d(attribute="test")


    def test_plotAttributeFeature1dMean(self):
        self.plot.load(self.data_file_path)

        self.plot.plotAttributeFeature1d(feature="directComparison",
                                         attribute="E",
                                         attribute_name="mean")

        self.compare_plot("directComparison_mean")


    def test_plotAttributeFeature1dVariance(self):
        self.plot.load(self.data_file_path)

        self.plot.plotAttributeFeature1d(feature="directComparison",
                                         attribute="Var",
                                         attribute_name="variance")

        self.compare_plot("directComparison_variance")



    def test_plotMeanError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.plotMean(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotMean(feature="feature2d")


    def test_plotMeanDirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.plotMean(feature="directComparison")

        self.compare_plot("directComparison_mean")


    def test_plotMeanfeature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.plotMean(feature="feature1d")

        self.compare_plot("feature1d_mean")


    def test_plotVarianceError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.plotVariance(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotVariance(feature="feature2d")


    def test_plotVarianceDirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.plotVariance(feature="directComparison")

        self.compare_plot("directComparison_variance")


    def test_plotVariancefeature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.plotVariance(feature="feature1d")

        self.compare_plot("feature1d_variance")



    def test_plotMeanAndVarianceError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.plotMeanAndVariance(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotMeanAndVariance(feature="feature2d")


    def test_plotMeanAndVarianceDirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.plotMeanAndVariance(feature="directComparison")

        self.compare_plot("directComparison_mean-variance")


    def test_plotMeanAndVariancefeature1d(self):
        self.plot.load(self.data_file_path)

        self.plot.plotMeanAndVariance(feature="feature1d")

        self.compare_plot("feature1d_mean-variance")



    def test_plotConfidenceIntervalDirectComparison(self):
        self.plot.load(self.data_file_path)

        self.plot.plotConfidenceInterval(feature="directComparison")

        self.compare_plot("directComparison_confidence-interval")


    def test_plotConfidenceIntervalFeature0d(self):
        self.plot.load(self.data_file_path)

        self.plot.plotConfidenceInterval(feature="feature1d")

        self.compare_plot("feature1d_confidence-interval")


    def test_plotConfidenceIntervalError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.plotConfidenceInterval(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotConfidenceInterval(feature="feature2d")


    def test_plotSensitivityDirectComparison1(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivity(feature="directComparison", sensitivity="sensitivity_1")

        self.compare_plot("directComparison_sensitivity_1_a")
        self.compare_plot("directComparison_sensitivity_1_b")


    def test_plotSensitivityDirectComparisonT(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivity(feature="directComparison", sensitivity="sensitivity_t")

        self.compare_plot("directComparison_sensitivity_t_a")
        self.compare_plot("directComparison_sensitivity_t_b")


    def test_plotSensitivityFeature1d1(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivity(feature="feature1d", sensitivity="sensitivity_1")


        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")


    def test_plotSensitivityFeature1dt(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivity(feature="feature1d", sensitivity="sensitivity_t")


        self.compare_plot("feature1d_sensitivity_t_a")
        self.compare_plot("feature1d_sensitivity_t_b")


    def test_plotSensitivityError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.plotSensitivity(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotSensitivity(feature="feature2d")



    def test_plotSensitivityCombinedDirectComparison1(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivityCombined(feature="directComparison", sensitivity="sensitivity_1")

        self.compare_plot("directComparison_sensitivity_1")


    def test_plotSensitivityCombinedDirectComparisonT(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivityCombined(feature="directComparison", sensitivity="sensitivity_t")

        self.compare_plot("directComparison_sensitivity_t")


    def test_plotSensitivityCombinedFeature1d1(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivityCombined(feature="feature1d", sensitivity="sensitivity_1")

        self.compare_plot("feature1d_sensitivity_1")

    def test_plotSensitivityCombinedFeature1dT(self):
        self.plot.load(self.data_file_path)

        self.plot.plotSensitivityCombined(feature="feature1d", sensitivity="sensitivity_t")

        self.compare_plot("feature1d_sensitivity_t")


    def test_plotSensitivityCombinedError(self):
        self.plot.load(self.data_file_path)

        with self.assertRaises(ValueError):
            self.plot.plotSensitivityCombined(feature="feature0d")

        with self.assertRaises(ValueError):
            self.plot.plotSensitivityCombined(feature="feature2d")



    def test_plot1dFeatures(self):
        self.plot.load(self.data_file_path)

        self.plot.plot1dFeatures(sensitivity="sensitivity_1")

        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_a")
        self.compare_plot("feature1d_sensitivity_1_b")
        self.compare_plot("feature1d_sensitivity_1")



    def test_plot0dFeature(self):
        self.plot.load(self.data_file_path)

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_1")

        self.compare_plot("feature0d_sensitivity_1")

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_t")

        self.compare_plot("feature0d_sensitivity_t")


        with self.assertRaises(ValueError):
            self.plot.plot0dFeature(feature="feature1d")


    def test_plot0dFeatures1(self):
        self.plot.load(self.data_file_path)

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_1")

        self.compare_plot("feature0d_sensitivity_1")



    def test_plot0dFeaturesT(self):
        self.plot.load(self.data_file_path)

        self.plot.plot0dFeature(feature="feature0d", sensitivity="sensitivity_t")

        self.compare_plot("feature0d_sensitivity_t")


    def test_plotCondensed(self):
        self.plot.load(self.data_file_path)

        self.plot.plotCondensed(sensitivity="sensitivity_1")

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")



    def test_plotCondensedNoSensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plotCondensedNoSensitivity()

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature0d_sensitivity_1")



    def test_plotAllData1(self):
        self.plot.load(self.data_file_path)

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


        self.compare_plot("total-sensitivity_1_grid")

    def test_plotAllDataT(self):
        self.plot.load(self.data_file_path)

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


        self.compare_plot("total-sensitivity_t_grid")


    def test_plotAllDataNoSensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plotAllDataNoSensitivity()

        self.compare_plot("directComparison_mean")
        self.compare_plot("directComparison_variance")
        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")




    def test_plotAllDataAllSensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plotAllDataAllSensitivity()

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



    def test_plotPlotCondensed(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=True, sensitivity=True)

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")


    def test_plotPlotCondensedNoSensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=True, sensitivity=False)

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")

        self.compare_plot("feature0d_sensitivity_1")



    def test_plotPlotAllDataNoSensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=False, sensitivity=False)

        self.compare_plot("directComparison_mean")
        self.compare_plot("directComparison_variance")
        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")


        self.compare_plot("feature1d_mean")
        self.compare_plot("feature1d_variance")
        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")



    def test_plotPlotAllDataAllSensitivity(self):
        self.plot.load(self.data_file_path)

        self.plot.plot(condensed=False, sensitivity=True)

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

# TODO test plotAllDataFromExploration
# TODO test plotAllDataInFolder



if __name__ == "__main__":
    unittest.main()
