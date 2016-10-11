import subprocess
import datetime
import uncertainpy

parameterlist = [["kappa", -0.1, None],
                 ["gamma", 0.1, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.CoffeeCupPointModelDependent(parameters)
# This sets all distributions to the same, not necessary for exploreParameters,
# but necessary for compareMC
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                feature_list=None,
                                                save_figures=True,
                                                output_dir_data="../../uncertainpy_results/data/coffee_dependent_single",
                                                output_dir_figures="../../uncertainpy_results/figures/coffee_dependent_single",
                                                rosenblatt=True)


uncertainty.allParameters()


subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(uncertainty.timePassed())))
