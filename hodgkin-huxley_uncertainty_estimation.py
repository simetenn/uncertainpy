import subprocess
import datetime

import uncertainpy


data_dir = "data/"
output_figures_dir = "figures/"
figureformat = ".png"
output_gif_dir = "gifs/"


memory = uncertainpy.Memory(10)
memory.start()

parameterlist = [["V_rest", 0, None],
                 ["gbar_Na", 120, None],
                 ["Cm", 1, None],
                 ["gbar_K", 36, None],
                 ["gbar_l", 0.3, None],
                 ["E_Na", 115, None],
                 ["E_K", -12], None,
                 ["E_l", 10.613, None]]

parameterlist = [["gbar_Na", 120, None],
                 ["gbar_K", 36, None],
                 ["gbar_l", 0.3, None]]


parameters = uncertainpy.Parameters(parameterlist)

model = uncertainpy.HodkinHuxleyModel(parameters)

# percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
percentages = [0.05]
test_distributions = {"uniform": percentages}
exploration = uncertainpy.UncertaintyEstimations(model, test_distributions, features="all", CPUs=1,
                                                 output_dir_data="data/hodgkin-huxley")
exploration.exploreParameters()

plot = uncertainpy.PlotUncertainty(data_dir="data/hodgkin-huxley", output_figures_dir="figures/hodgkin-huxley")
plot.plotAllData()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
