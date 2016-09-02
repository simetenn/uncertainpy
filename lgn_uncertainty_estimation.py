import subprocess
import datetime

import uncertainpy

model_file = "INmodel.hoc"
model_path = "uncertainpy/models/neuron_models/dLGN_modelDB/"


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


parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.NeuronModel(parameters=parameters,
                                model_file=model_file,
                                model_path=model_path,
                                adaptive_model=True)
model.setAllDistributions(uncertainpy.Distribution(0.05).uniform)


exploration = uncertainpy.UncertaintyEstimations(model, CPUs=6, supress_model_output=True,
                                                 feature_list="all",
                                                 output_dir_data="data/lgn",
                                                 save_figures=False,
                                                 output_dir_figures="figures/lgn")


# mc_samples = [10, 100, 1000]
# exploration.compareMC(mc_samples)

#distributions = {"uniform": np.linspace(0.01, 0.1, 10), "normal": np.linspace(0.01, 0.1, 10)}
percentages = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# percentages = [0.02, 0.03, 0.04]
distributions = {"uniform": percentages}
exploration.exploreParameters(distributions)


subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
