import subprocess
import datetime

from memory import Memory
from uncertainty import UncertaintyEstimations, UncertaintyEstimation
from distribution import Distribution
from parameters import Parameters
from NeuronModel import NeuronModel

modelfile = "INmodel.hoc"
modelpath = "neuron_models/dLGN_modelDB/"
parameterfile = "Parameters.hoc"

data_dir = "data/"
output_figures_dir = "figures/"
figureformat = ".png"
output_gif_dir = "gifs/"

original_parameters = {
    "rall": 113,       # Taken from litterature
    "cap": 1.1,        #
    "Rm": 22000,       # Estimated by hand
    "Vrest": -63,      # Experimentally measured
    "Epas": -67,       # Estimated by hand
    "gna": 0.09,
    "nash": -52.6,
    "gkdr": 0.37,
    "kdrsh": -51.2,
    "gahp": 6.4e-5,
    "gcat": 1.17e-5,    # Estimatedmodel
}

memory = Memory(10)
memory.start()

fitted_parameters = ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat", "gcal",
                     "ghbar", "catau", "gcanbar"]

test_parameters = ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat"]
test_parameters = ["Rm", "Epas"]

distribution_function = Distribution(0.1).uniform
distribution_functions = {"Rm": distribution_function, "Epas": distribution_function}



#parameters = Parameters(original_parameters, distribution_function, test_parameters)
parameters = Parameters(original_parameters, distribution_function, fitted_parameters)

model = NeuronModel(modelfile, modelpath)

test_distributions = {"uniform": [0.05, 0.06], "normal": [0.04, 0.05]}
#test_distributions = {"uniform": np.linspace(0.01, 0.1, 2)}



#
# percentages = np.linspace(0.01, 0.1, 41)[23:]
percentages = [0.02, 0.03]
test_distributions = {"uniform": percentages}
exploration = UncertaintyEstimations(model, original_parameters, test_parameters, test_distributions,
                                     output_dir_data="data/test", CPUs=2)
exploration.exploreParameters()

#distributions = {"uniform": np.linspace(0.01, 0.1, 10), "normal": np.linspace(0.01, 0.1, 10)}
#percentages = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
# # percentages = [0.02, 0.03, 0.04, 0.05]
# distributions = {"uniform": percentages}
# exploration = UncertaintyEstimations(model, fitted_parameters, distributions)
# exploration.exploreParameters()

# distributions = {"normal": percentages}
# exploration = UncertaintyEstimations(model, fitted_parameters, distributions)
# exploration.exploreParameters()
memory.end()



# plot = PlotUncertainty(data_dir=data_dir,
#                        output_figures_dir=output_figures_dir,
#                        figureformat=figureformat,
#                        output_gif_dir=output_gif_dir)
#
# plot.allData()
# plot.gif()
# sortByParameters()


subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
