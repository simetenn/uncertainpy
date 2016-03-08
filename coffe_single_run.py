import subprocess
import datetime
import uncertainpy

memory = uncertainpy.Memory(1)
memory.startTotal()

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.CoffeeCupPointModel(parameters)
# This sets all distributions to the same, not necessary for exploreParameters,
# but necessary for compareMC
model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                CPUs=1,
                                                supress_model_output=True,
                                                feature_list=None,
                                                save_figures=True,
                                                output_dir_data="data/coffee",
                                                output_dir_figures="figures/coffee",
                                                rosenblatt=False)


uncertainty.singleParameters()
uncertainty.allParameters()
# uncertainty.plotSimulatorResults()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(uncertainty.timePassed())))
