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
