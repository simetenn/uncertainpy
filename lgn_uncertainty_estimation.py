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



memory = uncertainpy.Memory(10)
memory.start()

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.NeuronModel(parameters=parameters, model_file=model_file, model_path=model_path)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


exploration = uncertainpy.UncertaintyEstimations(model, CPUs=7, supress_model_output=True,
                                                 feature_list="all",
                                                 output_dir_data="data/lgn",
                                                 save_figures=True,
                                                 output_dir_figures="figures/lgn")

#distributions = {"uniform": np.linspace(0.01, 0.1, 10), "normal": np.linspace(0.01, 0.1, 10)}
percentages = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
distributions = {"uniform": percentages}
exploration.exploreParameters(distributions)

mc_samples = [50, 100, 200, 500, 1000, 1500, 2000]
exploration.compareMC(mc_samples)


memory.end()


subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
