import numpy as np
import unittest
import chaospy as cp
import os
import shutil
import subprocess


from uncertainpy import UncertaintyEstimation
from uncertainpy.parameters import Parameters
from uncertainpy.features import GeneralFeatures
from uncertainpy import Distribution

from features import TestingFeatures
from models import TestingModel0d, TestingModel1d, TestingModel2d
from models import TestingModel1dAdaptive


class TestUncertainty(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


        self.uncertainty = UncertaintyEstimation(TestingModel1d(),
                                                 features=TestingFeatures(),
                                                 verbose_level="error",
                                                 seed=self.seed)



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



    def test_resetValues(self):
        self.uncertainty.parameter_names = -1
        self.uncertainty.parameter_space = -1

        self.uncertainty.U_mc = -1
        self.uncertainty.U_hat = -1
        self.uncertainty.distribution = -1
        self.uncertainty.P = -1


        self.uncertainty.data.U = -1
        self.uncertainty.solves = -1
        self.uncertainty.data.t = -1
        self.uncertainty.data.E = -1
        self.uncertainty.data.Var = -1
        self.uncertainty.data.p_05 = -1
        self.uncertainty.data.p_95 = -1
        self.uncertainty.data.sensitivity_1 = -1
        self.uncertainty.data.total_sensitivity_1 = -1
        self.uncertainty.data.sensitivity_t = -1
        self.uncertainty.data.total_sensitivity_t = -1


        self.uncertainty.resetValues()

        self.assertIsNone(self.uncertainty.parameter_names)
        self.assertIsNone(self.uncertainty.parameter_space)
        self.assertEqual(self.uncertainty.data.U, {})
        self.assertEqual(self.uncertainty.U_hat, {})
        self.assertIsNone(self.uncertainty.distribution)
        self.assertIsNone(self.uncertainty.solves)
        self.assertEqual(self.uncertainty.data.t, {})
        self.assertEqual(self.uncertainty.data.E, {})
        self.assertEqual(self.uncertainty.data.Var, {})
        self.assertEqual(self.uncertainty.U_mc, {})
        self.assertEqual(self.uncertainty.data.p_05, {})
        self.assertEqual(self.uncertainty.data.p_95, {})
        self.assertEqual(self.uncertainty.data.sensitivity_1, {})
        self.assertEqual(self.uncertainty.data.total_sensitivity_1, {})
        self.assertEqual(self.uncertainty.data.sensitivity_t, {})
        self.assertEqual(self.uncertainty.data.total_sensitivity_t, {})
        self.assertIsNone(self.uncertainty.P)







    def test_createPCExpansion(self):

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)


        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(features_to_run=["feature0d",
                                                                                           "feature1d",
                                                                                           "feature2d"]),
                                                 verbose_level="error")
        self.uncertainty.createPCExpansion()

        self.assertIsInstance(self.uncertainty.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["directComparison"], cp.Poly)


    def test_createPCExpansionAdaptiveError(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1dAdaptive(parameters, adaptive_model=False)
        model.setAllDistributions(Distribution(0.5).uniform)


        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(features_to_run=["feature1d",
                                                                                           "feature2d"]),
                                                 verbose_level="error")
        with self.assertRaises(ValueError):
            self.uncertainty.createPCExpansion()


    def test_createPCExpansionFeatureInvalid(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.mkdir(self.output_test_dir)

        logfile = os.path.join(self.output_test_dir, "test.log")

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(features_to_run="featureInvalid"),
                                                 verbose_level="warning",
                                                 verbose_filename=logfile)

        self.uncertainty.createPCExpansion()
        self.assertIsInstance(self.uncertainty.U_hat["directComparison"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["featureInvalid"], cp.Poly)

        message = "WARNING - uncertainty - Feature: featureInvalid does not yield results for all parameter combinations"
        self.assertTrue(message in open(logfile).read())

        shutil.rmtree(self.output_test_dir)


    def test_createPCExpansionRosenBlatt(self):

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)


        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(),
                                                 rosenblatt=True,
                                                 verbose_level="error")

        self.uncertainty.createPCExpansion()


        self.assertIsInstance(self.uncertainty.U_hat["feature0d"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["feature1d"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["feature2d"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["directComparison"], cp.Poly)
        self.assertIsInstance(self.uncertainty.U_hat["featureInvalid"], cp.Poly)






    

    def test_singleParameters(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.mkdir(self.output_test_dir)

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)


        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(),
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed)


        self.uncertainty.singleParameters()


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-a.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-a.h5")
        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])


        self.assertEqual(result, 0)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-b.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-b.h5")
        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])

        self.assertEqual(result, 0)





    def test_allParameters(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.mkdir(self.output_test_dir)

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]



        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(),
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error")


        self.uncertainty.allParameters()

        # filename = os.path.join(self.output_test_dir, "test_save_data")
        # self.assertTrue(os.path.isfile(filename))

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d.h5")

        # TODO find out why this is needed for different machines
        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])

        self.assertEqual(result, 0)


    def test_singleParametersMC(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.mkdir(self.output_test_dir)

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]



        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(),
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_data_filename="TestingModel1d_MC",
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error",
                                                 nr_mc_samples=10**1)


        self.uncertainty.singleParametersMC()

        # self.assertTrue(os.path.isfile(os.path.join(self.output_test_dir,
        #                                "test_save_data_MC_single-parameter-a")))
        # self.assertTrue(os.path.isfile(os.path.join(self.output_test_dir,
        #                                "test_save_data_MC_single-parameter-b")))

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC_single-parameter-a.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC_single-parameter-a.h5")

        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])

        self.assertEqual(result, 0)

        compare_file = os.path.join(folder, "data/TestingModel1d_MC_single-parameter-b.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC_single-parameter-b.h5")

        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])

        self.assertEqual(result, 0)


    def test_allParametersMC(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.mkdir(self.output_test_dir)

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]



        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(),
                                                 save_data=True,
                                                 save_figures=False,
                                                 output_data_filename="TestingModel1d_MC",
                                                 output_dir_data=self.output_test_dir,
                                                 verbose_level="error",
                                                 nr_mc_samples=10**1)


        self.uncertainty.allParametersMC()

        # filename = os.path.join(self.output_test_dir, "test_save_data_MC")
        # self.assertTrue(os.path.isfile(filename))

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_MC.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_MC.h5")

        result = subprocess.call(["h5diff", "-d", "1e-10", filename, compare_file])

        self.assertEqual(result, 0)


    def test_plotAll(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(),
                                                 save_data=False,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed)


        self.uncertainty.allParameters()
        self.uncertainty.plotAll()

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
        self.compare_plot("total-sensitivity_1_grid")





    def test_plotResults(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=TestingFeatures(),
                                                 save_data=False,
                                                 save_figures=False,
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
                                                 verbose_level="error",
                                                 seed=self.seed)


        self.uncertainty.allParameters()
        self.uncertainty.plotResults()

        self.compare_plot("directComparison_mean-variance")
        self.compare_plot("directComparison_confidence-interval")
        self.compare_plot("directComparison_sensitivity_1_grid")

        self.compare_plot("feature1d_mean-variance")
        self.compare_plot("feature1d_confidence-interval")
        self.compare_plot("feature1d_sensitivity_1_grid")

        self.compare_plot("feature0d_sensitivity_1")

        self.compare_plot("featureInvalid_sensitivity_1")

        self.compare_plot("total-sensitivity_1_grid")









    def compare_plot(self, name):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "figures/TestingModel1d",
                                    name + ".png")

        plot_file = os.path.join(self.output_test_dir,
                                 name + ".png")

        result = subprocess.call(["diff", plot_file, compare_file])

        self.assertEqual(result, 0)



if __name__ == "__main__":
    unittest.main()
