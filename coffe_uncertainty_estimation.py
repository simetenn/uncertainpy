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


exploration = uncertainpy.UncertaintyEstimations(model,
                                                 feature_list=None,
                                                 save_figures=True,
                                                 output_dir_data="data/coffee",
                                                 output_dir_figures="figures/coffee",
                                                 plot_simulator_results=False,
                                                 rosenblatt=False)



# percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# percentages = range(1, 100, 10)
# print percentages
# test_distributions = {"uniform": percentages[::-1]}
# exploration.exploreParameters(test_distributions)

# mc_samples = [10, 100, 1000, 10000, 100000, 1000000]
mc_samples = [10, 100, 1000]
exploration.compareMC(mc_samples)

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
