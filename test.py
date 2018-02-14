import unittest
import sys
import click
import collections

import matplotlib
matplotlib.use('Agg')

from tests import *


verbose = 1



def create_test_suite_parameter(testcase, parameter=False):
    """
Create a suite containing all tests taken from the given
class, passing them a parameter.
    """
    loader = unittest.TestLoader()
    testnames = loader.getTestCaseNames(testcase)
    suite = unittest.TestSuite()
    for name in testnames:
        suite.addTest(testcase(name, parameter=parameter))
    return suite


def to_iterable(iterable):
    if not isinstance(iterable, collections.Iterable):
        iterable = [iterable]
    return iterable


def create_test_suite(test_classes_to_run=[], parameter_test_cases=[], parameter=None):
    loader = unittest.TestLoader()

    test_classes_to_run = to_iterable(test_classes_to_run)
    parameter_test_cases = to_iterable(parameter_test_cases)


    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    for test_class in parameter_test_cases:
        suite = create_test_suite_parameter(test_class, parameter=parameter)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    return big_suite


def run(test_cases=[], parameter_test_cases=[], parameter=None):
    suite = create_test_suite(test_cases, parameter_test_cases, parameter)

    runner = unittest.TextTestRunner(verbosity=verbose)
    results = runner.run(suite)

    errors = len(results.errors)
    failures = len(results.failures)
    run = results.testsRun
    print("------------------------------------------------------")
    print("Test: run={} errors={} failures={}".format(run, errors, failures))

    if not results.wasSuccessful():
        sys.exit(1)



testing_spikes = [TestSpike, TestSpikes]

testing_features = [TestFeatures, TestGeneralSpikingFeatures, TestSpikingFeatures,
                    TestTestingFeatures, TestNetworkFeatures, TestGeneralNetworkFeatures,
                    TestEfelFeatures]

testing_base = [TestBase, TestParameterBase]

testing_data = [TestData, TestDataFeature]

# TODO: several tests crashes when several tests with Xvfb is run one after another
testing_models = [TestTestingModel0d, TestTestingModel1d, TestTestingModel2d,
                  TestModel, TestHodgkinHuxleyModel, TestCoffeeCupModel,
                  TestIzhikevichModel, TestNestModel, TestNeuronModel,
                  TestRunModel, TestParallel]

testing_parameters = [TestParameter, TestParameters]

testing_exact = testing_spikes + [TestUncertainty, TestPlotUncertainpy]

testing_all = testing_parameters + testing_models + testing_base\
                   + testing_features + testing_data

testing_complete = testing_all + [TestExamples]



@click.group()
@click.option('--verbosity', default=1, help="Verbosity of test runner.")
def cli(verbosity):
    global verbose
    verbose = verbosity


@cli.command()
def distribution():
    run(TestDistribution)


@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def spike(exact):
    run(parameter_test_cases=TestSpike, parameter=exact)


@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def spikes(exact):
    run(parameter_test_cases=TestSpikes, parameter=exact)


@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def all_spikes(exact):
    run(parameter_test_cases=spikes, parameter=exact)


@cli.command()
def features():
    run(TestFeatures)

@cli.command()
def general_spiking_features():
    run(TestGeneralSpikingFeatures)


@cli.command()
def spiking_features():
    run(TestSpikingFeatures)


@cli.command()
def test_features():
    run(TestTestingFeatures)


@cli.command()
def efel_features():
    run(TestEfelFeatures)


@cli.command()
def network_features():
    run(TestNetworkFeatures)


@cli.command()
def general_network_features():
    run(TestGeneralNetworkFeatures)


@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def all_features(exact):
    run(testing_features, testing_spikes, exact)


@cli.command()
def logger():
    run(TestLogger)


@cli.command()
def uncertainty_calculations():
    run(TestUncertaintyCalculations)

@cli.command()
def base():
    run(TestBase)


@cli.command()
def parameter_base():
    run(TestParameterBase)


@cli.command()
def all_bases():
    run(testing_base)


@cli.command()
def parameter():
    run(TestParameter)


@cli.command()
def parameters():
    run(TestParameters)


@cli.command()
def all_parameters():
    run(testing_parameters)


@cli.command()
def data_feature():
    run(TestDataFeature)


@cli.command()
def data():
    run(TestData)


@cli.command()
def all_data():
    run(testing_data)

@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def plotting(exact):
    run(parameter_test_cases=TestPlotUncertainpy, parameter=exact)


@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def uncertainty(exact):
    run(parameter_test_cases=TestUncertainty, parameter=exact)


@cli.command()
def parallel():
    run(TestParallel)


@cli.command()
def run_model():
    run(TestRunModel)


@cli.command()
def model():
    run(TestModel)


@cli.command()
def nest_model():
    run(TestNestModel)

@cli.command()
def neuron_model():
    run(TestNeuronModel)


@cli.command()
def models():
    run(testing_models)


@cli.command()
def examples():
    run(TestExamples)


@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def all(exact):
    run(testing_all, testing_exact, exact)


@cli.command()
@click.option('--exact', default=False, is_flag=True,
              help="Test if the plot files are exactly equal.")
def complete(exact):
    run(testing_complete, testing_exact, exact)

if __name__ == '__main__':
    cli()

