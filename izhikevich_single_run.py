import subprocess
import datetime
import uncertainpy

memory = uncertainpy.utils.Memory(10)
memory.start()

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.models.IzhikevichModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(1).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                CPUs=8,
                                                save_figures=True,
                                                feature_list="all",
                                                output_dir_data="data/izhikevich",
                                                output_dir_figures="figures/izhikevich",
                                                nr_mc_samples=10**2,
                                                combined_features=True)


uncertainty.singleParameters()
uncertainty.allParameters()
# uncertainty.plotSimulatorResults()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(uncertainty.timePassed())))
