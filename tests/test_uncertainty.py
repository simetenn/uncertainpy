import numpy as np
import unittest
import scipy.interpolate
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
                                                 output_dir_data=self.output_test_dir,
                                                 output_dir_figures=self.output_test_dir,
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



    def test_PCAnalysis(self):
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


        self.uncertainty.data.feature_list = ["feature0d", "feature1d",
                                              "feature2d", "directComparison"]

        q0, q1 = cp.variable(2)
        parameter_space = model.parameters.getUncertain("parameter_space")
        self.uncertainty.distribution = cp.J(*parameter_space)

        self.uncertainty.data.uncertain_parameters = ["a", "b"]


        self.uncertainty.U_hat["directComparison"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty.U_hat["feature0d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty.U_hat["feature1d"] = cp.Poly([q0, q1*q0, q1])
        self.uncertainty.U_hat["feature2d"] = cp.Poly([q0, q1*q0, q1])

        self.uncertainty.PCAnalysis()


        # Test if all calculated properties actually exists
        self.assertIsInstance(self.uncertainty.data.p_05["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.p_05["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.p_05["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.p_05["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty.data.p_95["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.p_95["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.p_95["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.p_95["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty.data.E["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.E["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.E["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.E["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty.data.Var["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.Var["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.Var["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.Var["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty.data.sensitivity_1["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.sensitivity_1["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.sensitivity_1["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.sensitivity_1["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty.data.total_sensitivity_1["directComparison"], list)
        self.assertIsInstance(self.uncertainty.data.total_sensitivity_1["feature0d"], list)
        self.assertIsInstance(self.uncertainty.data.total_sensitivity_1["feature1d"], list)
        self.assertIsInstance(self.uncertainty.data.total_sensitivity_1["feature2d"], list)


        self.assertIsInstance(self.uncertainty.data.sensitivity_t["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.sensitivity_t["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.sensitivity_t["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.data.sensitivity_t["feature2d"], np.ndarray)

        self.assertIsInstance(self.uncertainty.data.total_sensitivity_t["directComparison"], list)
        self.assertIsInstance(self.uncertainty.data.total_sensitivity_t["feature0d"], list)
        self.assertIsInstance(self.uncertainty.data.total_sensitivity_t["feature1d"], list)
        self.assertIsInstance(self.uncertainty.data.total_sensitivity_t["feature2d"], list)



        self.assertIsInstance(self.uncertainty.U_mc["directComparison"], np.ndarray)
        self.assertIsInstance(self.uncertainty.U_mc["feature0d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.U_mc["feature1d"], np.ndarray)
        self.assertIsInstance(self.uncertainty.U_mc["feature2d"], np.ndarray)




    def test_MC(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)
        features = TestingFeatures(features_to_run=None)

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 nr_mc_samples=10**1,
                                                 verbose_level="error")


        self.uncertainty.all_features = ["directComparison"]

        self.uncertainty.MC()

        self.assertTrue(np.allclose(self.uncertainty.data.E["directComparison"],
                                    np.arange(0, 10) + 3, atol=0.1))
        self.assertTrue(np.allclose(self.uncertainty.data.Var["directComparison"],
                                    np.zeros(10), atol=0.1))
        self.assertTrue(np.all(np.less(self.uncertainty.data.p_05["directComparison"],
                                       np.arange(0, 10) + 3)))
        self.assertTrue(np.all(np.greater(self.uncertainty.data.p_95["directComparison"],
                                          np.arange(0, 10) + 3)))


    def test_MC_feature0d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)
        features = TestingFeatures(features_to_run=["feature0d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 nr_mc_samples=10**1,
                                                 verbose_level="error")


        self.uncertainty.all_features = ["feature0d"]

        self.uncertainty.MC()

        self.assertTrue(np.array_equal(self.uncertainty.data.E["feature0d"],
                                       features.feature0d()))
        self.assertEqual(self.uncertainty.data.Var["feature0d"], 0)
        self.assertTrue(np.array_equal(self.uncertainty.data.p_05["feature0d"],
                                       features.feature0d()))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_95["feature0d"],
                                       features.feature0d()))


    def test_MC_feature1d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)
        features = TestingFeatures(features_to_run=["feature1d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 nr_mc_samples=10**1,
                                                 verbose_level="error")


        self.uncertainty.all_features = ["feature1d"]

        self.uncertainty.MC()

        self.assertTrue(np.array_equal(self.uncertainty.data.E["feature1d"],
                                       features.feature1d()))
        self.assertTrue(np.array_equal(self.uncertainty.data.Var["feature1d"],
                                       np.zeros(10)))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_05["feature1d"],
                                       features.feature1d()))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_95["feature1d"],
                                       features.feature1d()))



    def test_MC_feature2d(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)
        features = TestingFeatures(features_to_run=["feature2d"])

        self.uncertainty = UncertaintyEstimation(model,
                                                 features=features,
                                                 nr_mc_samples=10**1,
                                                 verbose_level="error")


        self.uncertainty.all_features = ["feature2d"]

        self.uncertainty.MC()

        self.assertTrue(np.array_equal(self.uncertainty.data.E["feature2d"],
                                       features.feature2d()))
        self.assertTrue(np.array_equal(self.uncertainty.data.Var["feature2d"],
                                       np.zeros((2, 10))))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_05["feature2d"],
                                       features.feature2d()))
        self.assertTrue(np.array_equal(self.uncertainty.data.p_95["feature2d"],
                                       features.feature2d()))



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
        result = subprocess.call(["h5diff",  "-d", "1e-10", filename, compare_file])


        self.assertEqual(result, 0)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d_single-parameter-b.h5")
        filename = os.path.join(self.output_test_dir, "TestingModel1d_single-parameter-b.h5")
        result = subprocess.call(["h5diff",  "-d", "1e-10", filename, compare_file])

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


def test_createMaskDirectComparison(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    masked_nodes, masked_U = self.runmodel.createMask(nodes, "directComparison")

    self.assertEqual(len(masked_U), 3)
    self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
    self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 3))
    self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10) + 5))
    self.assertTrue(np.array_equal(nodes, masked_nodes))


def test_createMaskDirectComparisonNaN(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    self.runmodel.data.U["directComparison"][1] = np.nan

    masked_nodes, masked_U = self.runmodel.createMask(nodes, "directComparison")


    self.assertEqual(len(masked_U), 2)
    self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
    self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 5))
    self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


def test_createMaskWarning(self):
    if os.path.isdir(self.output_test_dir):
        shutil.rmtree(self.output_test_dir)
    os.mkdir(self.output_test_dir)

    logfile = os.path.join(self.output_test_dir, "test.log")

    self.runmodel = RunModel(TestingModel1d(),
                                             features=TestingFeatures(),
                                             verbose_level="warning",
                                             verbose_filename=logfile)

    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]
    self.runmodel.verbose_filename = logfile

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    self.runmodel.data.U["directComparison"][1] = np.nan

    self.runmodel.createMask(nodes, "directComparison")

    message = "WARNING - uncertainty - Feature: directComparison does not yield results for all parameter combinations"
    self.assertTrue(message in open(logfile).read())

    shutil.rmtree(self.output_test_dir)


def test_createMaskFeature0d(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature0d")
    self.assertIn("feature0d", self.runmodel.data.U.keys())
    self.assertEqual(masked_U[0], 1)
    self.assertEqual(masked_U[1], 1)
    self.assertEqual(masked_U[2], 1)
    self.assertTrue(np.array_equal(nodes, masked_nodes))


def test_createMaskFeature0dNan(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    self.runmodel.data.U["feature0d"] = np.array([1, np.nan, 1])
    masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature0d")

    self.assertTrue(np.array_equal(masked_U, np.array([1, 1])))
    self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



def test_createMaskFeature1d(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    # feature1d
    masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature1d")
    self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10)))
    self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10)))
    self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10)))
    self.assertTrue(np.array_equal(nodes, masked_nodes))

    # feature2d

def test_createMaskFeature1dNan(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    self.runmodel.data.U["feature1d"] = np.array([np.arange(0, 10), np.nan, np.arange(0, 10)])
    masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature1d")

    self.assertTrue(np.array_equal(masked_U, np.array([np.arange(0, 10), np.arange(0, 10)])))
    self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



def test_createMaskFeature2d(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature2d")
    self.assertTrue(np.array_equal(masked_U[0],
                                   np.array([np.arange(0, 10),
                                             np.arange(0, 10)])))
    self.assertTrue(np.array_equal(masked_U[1],
                                   np.array([np.arange(0, 10),
                                             np.arange(0, 10)])))
    self.assertTrue(np.array_equal(masked_U[2],
                                   np.array([np.arange(0, 10),
                                             np.arange(0, 10)])))
    self.assertTrue(np.array_equal(nodes, masked_nodes))



def test_createMaskFeature2dNan(self):
    nodes = np.array([[0, 1, 2], [1, 2, 3]])
    self.runmodel.data.uncertain_parameters = ["a", "b"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    self.runmodel.data.U["feature2d"][1] = np.nan
    masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature2d")

    self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


    self.assertEqual(len(masked_U), 2)
    self.assertTrue(np.array_equal(masked_U[0],
                                   np.array([np.arange(0, 10),
                                             np.arange(0, 10)])))
    self.assertTrue(np.array_equal(masked_U[1],
                                   np.array([np.arange(0, 10),
                                             np.arange(0, 10)])))


def test_createMaskFeature2dnodes1DNaN(self):
    nodes = np.array([0, 1, 2])

    self.runmodel.data.uncertain_parameters = ["a"]

    results = self.runmodel.evaluateNodes(nodes)
    self.runmodel.storeResults(results)

    self.runmodel.data.U["feature2d"][1] = np.nan
    masked_nodes, masked_U = self.runmodel.createMask(nodes, "feature2d")

    self.assertTrue(np.array_equal(masked_nodes, np.array([0, 2])))

    self.assertEqual(len(masked_U), 2)
    self.assertTrue(np.array_equal(masked_U[0],
                                   np.array([np.arange(0, 10),
                                             np.arange(0, 10)])))
    self.assertTrue(np.array_equal(masked_U[1],
                                   np.array([np.arange(0, 10),
                                             np.arange(0, 10)])))




    def test_totalSensitivity1(self):
        self.runmodel.data.sensitivity_1 = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2]}
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        self.runmodel.totalSensitivity(sensitivity="sensitivity_1")

        self.assertEqual(self.runmodel.data.total_sensitivity_1["test2D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_1["test2D"][1], 2/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_1["test1D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_1["test1D"][1], 2/3.)


    def test_totalSensitivityT(self):
        self.runmodel.data.sensitivity_t = {"test2D": [[4, 6], [8, 12]], "test1D": [1, 2]}
        self.runmodel.data.uncertain_parameters = ["a", "b"]

        self.runmodel.totalSensitivity(sensitivity="sensitivity_t")

        self.assertEqual(self.runmodel.data.total_sensitivity_t["test2D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_t["test2D"][1], 2/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_t["test1D"][0], 1/3.)
        self.assertEqual(self.runmodel.data.total_sensitivity_t["test1D"][1], 2/3.)



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
