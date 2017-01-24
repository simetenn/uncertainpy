import numpy as np
import unittest
import scipy.interpolate
import chaospy as cp
import os
import shutil
import subprocess
import scipy.interpolate


from uncertainpy import UncertaintyCalculations
from uncertainpy.parameters import Parameters
from uncertainpy.features import GeneralFeatures
from uncertainpy import Distribution
from uncertainpy import RunModel

from features import TestingFeatures
from models import TestingModel0d, TestingModel1d, TestingModel2d
from models import TestingModel1dAdaptive


class TestUncertaintyCalculations(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=TestingFeatures(),
                                                                verbose_level="error",
                                                                seed=self.seed)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        UncertaintyCalculations(TestingModel1d())


    def test_intitFeatures(self):
        uncertainty = UncertaintyCalculations(TestingModel1d(),
                                              verbose_level="error")
        self.assertIsInstance(uncertainty.features, GeneralFeatures)

        uncertainty = UncertaintyCalculations(TestingModel1d(),
                                              features=TestingFeatures(),
                                              verbose_level="error")
        self.assertIsInstance(uncertainty.features, TestingFeatures)


    def test_createDistributionNone(self):

        self.uncertainty_calculations.createDistribution()

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)

    def test_createDistributionString(self):

        self.uncertainty_calculations.createDistribution("a")

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)

    def test_createDistributionList(self):

        self.uncertainty_calculations.createDistribution(["a"])

        self.assertIsInstance(self.uncertainty_calculations.distribution, cp.Dist)


    def test_createMaskDirectComparison(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "directComparison")

        self.assertEqual(len(masked_U), 3)
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(nodes, masked_nodes))


    def test_createMaskDirectComparisonNaN(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["directComparison"][1] = np.nan

        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "directComparison")


        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10) + 5))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))


    def test_createMaskWarning(self):
        logfile = os.path.join(self.output_test_dir, "test.log")

        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)
        model.setAllDistributions(Distribution(0.5).uniform)

        self.uncertainty_calculations = UncertaintyCalculations(model,
                                                                features=TestingFeatures(),
                                                                verbose_level="warning",
                                                                verbose_filename=logfile,
                                                                seed=self.seed)

        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["directComparison"][1] = np.nan

        self.uncertainty_calculations.createMask(nodes, "directComparison")

        message = "WARNING - uncertainty_calculations - Feature: directComparison does not yield results for all parameter combinations"
        self.assertTrue(message in open(logfile).read())


    def test_createMaskFeature0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature0d")

        self.assertEqual(masked_U[0], 1)
        self.assertEqual(masked_U[1], 1)
        self.assertEqual(masked_U[2], 1)
        self.assertTrue(np.array_equal(nodes, masked_nodes))


    def test_createMaskFeature0dNan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["feature0d"] = np.array([1, np.nan, 1])
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature0d")

        self.assertTrue(np.array_equal(masked_U, np.array([1, 1])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



    def test_createMaskFeature1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        # feature1d
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature1d")
        self.assertTrue(np.array_equal(masked_U[0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[1], np.arange(0, 10)))
        self.assertTrue(np.array_equal(masked_U[2], np.arange(0, 10)))
        self.assertTrue(np.array_equal(nodes, masked_nodes))

        # feature2d

    def test_createMaskFeature1dNan(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        self.uncertainty_calculations.data.U["feature1d"] = np.array([np.arange(0, 10), np.nan, np.arange(0, 10)])
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature1d")

        self.assertTrue(np.array_equal(masked_U, np.array([np.arange(0, 10), np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_nodes, np.array([[0, 2], [1, 3]])))



    def test_createMaskFeature2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature2d")
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
        uncertain_parameters = ["a", "b"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)

        self.uncertainty_calculations.data.U["feature2d"][1] = np.nan
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature2d")

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
        uncertain_parameters = ["a"]

        self.uncertainty_calculations.data = \
            self.uncertainty_calculations.runmodel.run(nodes, uncertain_parameters)


        self.uncertainty_calculations.data.U["feature2d"][1] = np.nan
        masked_nodes, masked_U = self.uncertainty_calculations.createMask(nodes, "feature2d")

        self.assertTrue(np.array_equal(masked_nodes, np.array([0, 2])))

        self.assertEqual(len(masked_U), 2)
        self.assertTrue(np.array_equal(masked_U[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(masked_U[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
