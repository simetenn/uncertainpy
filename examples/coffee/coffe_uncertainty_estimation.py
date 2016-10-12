import subprocess
import datetime
import uncertainpy
import numpy as np

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.CoffeeCupPointModel(parameters)
# This sets all distributions to the same, not necessary for exploreParameters,
# but necessary for compareMC
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


exploration = uncertainpy.UncertaintyEstimations(model,
                                                 feature_list=None,
                                                 save_figures=True,
                                                 output_dir_data="../../uncertainpy_results/data/coffee",
                                                 output_dir_figures="../../uncertainpy_results/figures/coffee",
                                                 plot_simulator_results=False,
                                                 rosenblatt=False)



percentages = np.linspace(0.1, 1, 10)
test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

mc_samples = [10, 100, 1000]
exploration.compareMC(mc_samples)


subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
