import subprocess
import datetime
import uncertainpy

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.models.IzhikevichModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(1).uniform)


exploration = uncertainpy.UncertaintyEstimations(model,
                                                 CPUs=8,
                                                 save_figures=True,
                                                 single_parameter_runs=True,
                                                 feature_list="all",
                                                 output_dir_data="../../uncertainpy_results/data/izhikevich",
                                                 output_dir_figures="../../uncertainpy_results/figures/izhikevich",
                                                 rosenblatt=False)

percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]

test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

mc_samples = [10, 100, 1000]
exploration.compareMC(mc_samples)


subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
