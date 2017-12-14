import unittest
import sys
import argparse

import matplotlib
matplotlib.use('Agg')


from tests import *

def create_test_suite(test_classes_to_run):
    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    return big_suite


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



parser = argparse.ArgumentParser(description="Run tests for Uncertainpy")
parser.add_argument("-u", "--utils", help="Utility tests", action="store_true")
parser.add_argument("-p", "--prerequisites",
                    help="Prerequisites tests",
                    action="store_true")
parser.add_argument("-a", "--all", help="All tests", action="store_true")
parser.add_argument("-b", "--basic",
                    help="Basic tests (all test up to Uncertainpy)",
                    action="store_true")
parser.add_argument("-f", "--fast", help="Run all tests except usecase test", action="store_true")
parser.add_argument("-e", "--exact",
                    help="Test if the plot files are exactly equal. " +\
                         "WARNING: uses diff, test fails on most machines " + \
                         "due to non vissible differences in the plots.",
                    action="store_true")


parser.add_argument("--uncertainpy", help="Uncertainpy tests", action="store_true")
# parser.add_argument("--exploration", help="UncertaintyQuantifications (explorations) test",
#                     action="store_true")
parser.add_argument("--parameters", help="Parameter tests", action="store_true")
parser.add_argument("--distribution", help="Distribution tests", action="store_true")
parser.add_argument("--spike", help="Spike tests", action="store_true")
parser.add_argument("--spikes", help="Spikes tests ", action="store_true")
parser.add_argument("--spike_sorting", help="Test spike sorting", action="store_true")
parser.add_argument("--plotuncertainty", help="PlotUncertainty tests", action="store_true")
# parser.add_argument("--plotuncertaintycompare", help="PlotUncertaintyCompare tests",
#                     action="store_true")
parser.add_argument("--features", help="Features tests", action="store_true")
parser.add_argument("--model", help="Model tests", action="store_true")
parser.add_argument("--runmodel", help="RunModel tests", action="store_true")
parser.add_argument("--logger", help="Logger tests", action="store_true")
parser.add_argument("--plotting", help="Plotting tests", action="store_true")
parser.add_argument("--parallel", help="Parallel tests", action="store_true")
parser.add_argument("--usecase", help="Usecase tests", action="store_true")
parser.add_argument("--data", help="Data tests", action="store_true")
parser.add_argument("--uncertaintycalculations", help="uncertaintyCalculations tests",
                    action="store_true")
parser.add_argument("--travis", help="Tests to run on travis", action="store_true")
parser.add_argument("--examples", help="Test all examples", action="store_true")


args = parser.parse_args()








test_distribution = create_test_suite([TestDistribution])

test_spike = create_test_suite_parameter(TestSpike, parameter=args.exact)
test_spikes = create_test_suite_parameter(TestSpikes, parameter=args.exact)
test_spike_sorting = create_test_suite([TestSpike, TestSpikes])

test_features = create_test_suite([TestFeatures,
                                   TestGeneralSpikingFeatures,
                                   TestSpikingFeatures,
                                   TestTestingFeatures,
                                   TestNetworkFeatures,
                                   TestGeneralNetworkFeatures,
                                   TestEfelFeatures])


test_features.addTest(test_spike_sorting)


test_logger = create_test_suite([TestLogger])

test_uncertaintycalculations = create_test_suite([TestUncertaintyCalculations])

test_base = create_test_suite([TestBase, TestParameterBase])

test_runModel = create_test_suite([TestRunModel])

test_parallel = create_test_suite([TestParallel])

test_model = create_test_suite([TestModel,
                                TestHodgkinHuxleyModel,
                                TestCoffeeCupModel,
                                TestIzhikevichModel,
                                TestTestingModel0d,
                                TestTestingModel1d,
                                TestTestingModel2d,
                                TestNeuronModel])
test_model.addTest(test_runModel)

test_parameters = create_test_suite([TestParameter, TestParameters])


test_data = create_test_suite([TestData, TestDataFeature])


test_plotUncertainty = create_test_suite_parameter(TestPlotUncertainpy, parameter=args.exact)

test_uncertainty = create_test_suite_parameter(TestUncertainty, parameter=args.exact)

test_usecase = create_test_suite([TestUseCases])

test_utils = unittest.TestSuite([test_logger])

test_plotting = unittest.TestSuite([test_plotUncertainty])

test_prerequisites = unittest.TestSuite([test_utils,
                                         test_parameters,
                                         test_distribution,
                                         test_features,
                                         test_model,
                                         test_data,
                                         test_base,
                                         test_parallel])


test_basic = unittest.TestSuite([test_prerequisites,
                                 test_uncertaintycalculations])

test_fast = unittest.TestSuite([test_basic, test_uncertainty, test_plotUncertainty])

test_examples = create_test_suite([TestExamples])

test_travis = unittest.TestSuite([test_basic, test_uncertainty, test_plotUncertainty])

test_all = unittest.TestSuite([test_basic, test_uncertainty,
                               test_plotUncertainty, test_plotUncertainty, test_examples])

# test_all = unittest.TestSuite([test_fast, test_usecase])
# test_all = unittest.TestSuite([test_fast, test_plotUncertaintyCompare, test_exploration])


test_runner = unittest.TextTestRunner()





results = {}

if args.utils:
    print("-----------------------------------------")
    print("Running testsuite: utils")
    results["utils"] = test_runner.run(test_utils)
if args.prerequisites:
    print("-----------------------------------------")
    print("Running testsuite: prerequisites")
    results["prerequisites"] = test_runner.run(test_prerequisites)
if args.basic:
    print("-----------------------------------------")
    print("Running testsuite: basic")
    results["basic"] = test_runner.run(test_basic)
if args.fast:
    print("-----------------------------------------")
    print("Running testsuite: fast")
    results["fast"] = test_runner.run(test_fast)
if args.uncertainpy:
    print("-----------------------------------------")
    print("Running testsuite: uncertainpy")
    results["uncertainpy"] = test_runner.run(test_uncertainty)
if args.parallel:
    print("-----------------------------------------")
    print("Running testsuite: parallel")
    results["parallel"] = test_runner.run(test_parallel)
# if args.exploration:
#     print "-----------------------------------------"
#     print "Running testsuite: exploration"
#     results["exploration"] = test_runner.run(test_exploration)
if args.parameters:
    print("-----------------------------------------")
    print("Running testsuite: parameters")
    results["parameters"] = test_runner.run(test_parameters)
if args.distribution:
    print("-----------------------------------------")
    print("Running testsuite: distribution")
    results["distribution"] = test_runner.run(test_distribution)
if args.spike_sorting:
    print("-----------------------------------------")
    print("Running testsuite: spike_sorting")
    results["spike_sorting"] = test_runner.run(test_spike_sorting)
if args.spike:
    print("-----------------------------------------")
    print("Running testsuite: spike")
    results["spike"] = test_runner.run(test_spike)
if args.spikes:
    print("-----------------------------------------")
    print("Running testsuite: spikes")
    results["spikes"] = test_runner.run(test_spikes)
if args.plotuncertainty:
    print("-----------------------------------------")
    print("Running testsuite: plotUncertainty")
    results["plotUncertainty"] = test_runner.run(test_plotUncertainty)
# if args.plotuncertaintycompare:
#     print "-----------------------------------------"
#     print "Running testsuite: plotUncertaintyCompare"
#     results["plotUncertaintyCompare"] = test_runner.run(test_plotUncertaintyCompare)
if args.features:
    print("-----------------------------------------")
    print("Running testsuite: features")
    results["features"] = test_runner.run(test_features)
if args.model:
    print("-----------------------------------------")
    print("Running testsuite: model")
    results["model"] = test_runner.run(test_model)
if args.runmodel:
    print("-----------------------------------------")
    print("Running testsuite: runModel")
    results["runModel"] = test_runner.run(test_runModel)
if args.logger:
    print("-----------------------------------------")
    print("Running testsuite: logger")
    results["logger"] = test_runner.run(test_logger)
if args.plotting:
    print("-----------------------------------------")
    print("Running testsuite: plotting")
    results["plotting"] = test_runner.run(test_plotting)
if args.usecase:
    print("-----------------------------------------")
    print("Running testsuite: usecase")
    results["usecase"] = test_runner.run(test_usecase)
if args.data:
    print("-----------------------------------------")
    print("Running testsuite: data")
    results["data"] = test_runner.run(test_data)
if args.uncertaintycalculations:
    print("-----------------------------------------")
    print("Running testsuite: uncertaintycalculations")
    results["uncertaintycalculations"] = test_runner.run(test_uncertaintycalculations)
if args.travis:
    print("-----------------------------------------")
    print("Running testsuite: travis")
    results["travis"] = test_runner.run(test_travis)
if args.all:
    print("-----------------------------------------")
    print("Running testsuite: all")
    results["all"] = test_runner.run(test_all)
if args.examples:
    print("-----------------------------------------")
    print("Running testsuite: examples")
    results["examples"] = test_runner.run(test_examples)


total_run = 0
total_errors = 0
total_failures = 0

print("----------------------------------------------------------------------")
print("             Test summary")
print("")
for key in results.keys():
    errors = len(results[key].errors)
    failures = len(results[key].failures)
    run = results[key].testsRun
    print("Test: {}, run={} errors={} failures={}".format(key, run, errors, failures))

    total_run += run
    total_errors += errors
    total_failures += failures

print("")
print("Total tests run={} errors={} failures={}".format(total_run, total_errors, total_failures))


for key in results.keys():
    if not results[key].wasSuccessful():
        sys.exit(1)
