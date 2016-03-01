import subprocess
import datetime
import uncertainpy

memory = uncertainpy.Memory(10)
memory.start()

# parameterlist = [["V_rest", 0, None],
#                  ["gbar_Na", 120, None],
#                  ["Cm", 1, None],
#                  ["gbar_K", 36, None],
#                  ["gbar_l", 0.3, None],
#                  ["E_Na", 115, None],
#                  ["E_K", -12], None,
#                  ["E_l", 10.613, None]]

parameterlist = [["gbar_Na", 120, None],
                 ["gbar_K", 36, None],
                 ["gbar_l", 0.3, None]]


parameters = uncertainpy.Parameters(parameterlist)

model = uncertainpy.HodkinHuxleyModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)

exploration = uncertainpy.UncertaintyEstimations(model,
                                                 feature_list="all",
                                                 CPUs=7,
                                                 output_dir_data="data/hodgkin-huxley",
                                                 output_dir_figures="figures/hodgkin-huxley",
                                                 save_figures=True,
                                                 rosenblatt=True)

# percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
# test_distributions = {"uniform": percentages}
# exploration.exploreParameters(test_distributions)

mc_samples = [10, 100, 1000]
exploration.compareMC(mc_samples)

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
