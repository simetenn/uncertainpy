import subprocess
import datetime
import uncertainpy

memory = uncertainpy.Memory(10)
memory.start()

parameterlist = [["kappa", -0.01, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.CoffeeCupPointModel(parameters)

# This sets all distributions to the same, not necessary for exploreParameters, but necessary for compareMC
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


exploration = uncertainpy.UncertaintyEstimations(model, output_dir_data="data/coffee")

percentages = [0.5]
test_distributions = {"uniform": percentages}

exploration.exploreParameters(test_distributions)


plot = uncertainpy.PlotUncertainty(data_dir="data/coffee", output_figures_dir="figures/coffee")
plot.plotAllDataFromExploration()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
