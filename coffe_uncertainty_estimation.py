import subprocess
import datetime
import uncertainpy


data_dir = "data/"
output_figures_dir = "figures/"
figureformat = ".png"
output_gif_dir = "gifs/"


memory = uncertainpy.Memory(10)
memory.start()


parameterlist = [["kappa", 0.02, None],
                 ["u_env", 0.2, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.CoffeeCupPointModel(parameters)

percentages = [0.1, 0.2]
test_distributions = {"uniform": percentages}

exploration = uncertainpy.UncertaintyEstimations(model,
                                                 test_distributions,
                                                 CPUs=1,
                                                 output_dir_data="data/coffee")
exploration.exploreParameters()
plot = uncertainpy.PlotUncertainty(data_dir="data/coffee", output_figures_dir="figures/coffee")
plot.plotAllData()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
