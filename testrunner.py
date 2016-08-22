import unittest
import os
import sys
import argparse

from uncertainpy.tests import *

def create_test_suite(test_classes_to_run):
    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    return big_suite


test_distribution = create_test_suite([TestDistribution])

test_features = create_test_suite([TestGeneralFeatures,
                                   TestGeneralNeuronFeatures,
                                   TestNeuronFeatures,
                                   TestTestingFeatures])

test_distribution = create_test_suite([TestDistribution])

test_logger = create_test_suite([TestLogger])

test_model = create_test_suite([TestModel, TestHodkinHuxleyModel, TestCoffeeCupPointModel,
                                TestIzhikevichModel, TestTestingModel0d, TestTestingModel1d,
                                TestTestingModel2d, TestTestingModel0dNoTime, TestTestingModel1dNoTime,
                                TestTestingModel2dNoTime, TestTestingModelNoU, TestNeuronModel])

test_parameters = create_test_suite([TestParameter, TestParameters])

test_plotting = create_test_suite([TestPrettyPlot, TestPrettyBar])

test_plotUncertainty = create_test_suite([TestPlotUncertainpy])

test_runModel = create_test_suite([TestRunModel])

test_spike_sorting =create_test_suite([TestSpike, TestSpikes])

test_uncertainty = create_test_suite([TestUncertainty])

test_exploration = create_test_suite([TestExploration])

test_usecase = create_test_suite([TestUseCases])

test_runner = unittest.TextTestRunner()


parser = argparse.ArgumentParser(description="Run a model simulation")


results = test_runner.run(test_model)
# print results
# print results.wasSuccessful()
#
# sys.exit(results.wasSuccessful())
