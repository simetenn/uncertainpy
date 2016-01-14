import subprocess
import datetime

import uncertainpy

model_file = "INmodel.hoc"
model_path = "uncertainpy/models/neuron_models/dLGN_modelDB/"

#
# distribution_function = uncertainpy.Distribution(0.1).uniform
# distribution_functions = {"Rm": distribution_function, "Epas": distribution_function}
#

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
model = uncertainpy.NeuronModel(parameters=parameters, model_file=model_file, model_path=model_path)

#test_distributions = {"uniform": [0.05, 0.06], "normal": [0.04, 0.05]}
#test_distributions = {"uniform": np.linspace(0.01, 0.1, 2)}
#
# percentages = np.linspace(0.01, 0.1, 11)
# #percentages = [0.02, 0.03]
# test_distributions = {"uniform": percentages}
# exploration = uncertainpy.UncertaintyEstimations(model, test_distributions, features="all", CPUs=1,
#                                                  output_dir_data="data/lgn")
#
#
# exploration.exploreParameters()

#distributions = {"uniform": np.linspace(0.01, 0.1, 10), "normal": np.linspace(0.01, 0.1, 10)}
# percentages = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
percentages = [0.2]
distributions = {"uniform": percentages}
exploration = uncertainpy.UncertaintyEstimations(model, distributions,supress_output=False,
                                                 CPUs=1,
                                                 feature_list=None,
                                                 output_dir_data="data/lgn")
exploration.exploreParameters()

plot = uncertainpy.PlotUncertainty(data_dir="data/lgn", output_figures_dir="figures/lgn")
plot.plotAllData()

memory.end()


subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
