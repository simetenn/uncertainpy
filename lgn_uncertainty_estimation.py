import subprocess
import datetime

import uncertainpy

modelfile = "INmodel.hoc"
modelpath = "uncertainpy/models/neuron_models/dLGN_modelDB/"
parameterfile = "Parameters.hoc"

data_dir = "data/"
output_figures_dir = "figures/"
figureformat = ".png"
output_gif_dir = "gifs/"

distribution_function = uncertainpy.Distribution(0.1).uniform
distribution_functions = {"Rm": distribution_function, "Epas": distribution_function}


parameterlist = [["cap", 1.1, None],
                 ["Rm", 22000, None],
                 ["Vrest", -63, None],
                 ["Epas", -67, None],
                 ["gna", 0.09, None],
                 ["nash", -52.6, None],
                 ["gkdr", 0.37, None],
                 ["kdrsh", -51.2, None],
                 ["gahp", 6.4e-5, None],
                 ["gcat", 1.17e-5, None]]


parameterlist = [["Rm", 22000, None],
                 ["Epas", -67, None]]

memory = uncertainpy.Memory(10)
memory.start()

#parameters = Parameters(original_parameters, distribution_function, test_parameters)
parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.NeuronModel(modelfile, modelpath, parameters)

#test_distributions = {"uniform": [0.05, 0.06], "normal": [0.04, 0.05]}
#test_distributions = {"uniform": np.linspace(0.01, 0.1, 2)}

# percentages = np.linspace(0.01, 0.1, 41)[23:]
percentages = [0.02, 0.03]
test_distributions = {"uniform": percentages}
exploration = uncertainpy.UncertaintyEstimations(model, test_distributions,
                                                 output_dir_data="data/test")
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
