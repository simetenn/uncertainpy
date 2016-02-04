import subprocess
import datetime
import uncertainpy

memory = uncertainpy.Memory(10)
memory.start()

parameterlist = [["kappa", -0.01, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.CoffeeCupPointModel(parameters)
# This sets all distributions to the same, not necessary for exploreParameters,
# but necessary for compareMC
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


exploration = uncertainpy.UncertaintyEstimations(model,
                                                 feature_list=None,
                                                 save_figures=True,
                                                 output_dir_data="data/coffee",
                                                 output_figures_dir="figures/coffee")



percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

mc_samples = [50, 100, 200, 500, 1000, 1500, 2000]
exploration.compareMC(mc_samples)

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
