import subprocess
import datetime
import uncertainpy

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.models.IzhikevichModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                CPUs=2,
                                                save_figures=True,
                                                feature_list="all",
                                                output_dir_data="../../uncertainpy_results/data/izhikevich",
                                                output_dir_figures="../../uncertainpy_results/figures/izhikevich",
                                                nr_mc_samples=10**2,
                                                combined_features=True)


uncertainty.singleParameters()
uncertainty.allParameters()
# uncertainty.plotSimulatorResults()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(uncertainty.timePassed())))
