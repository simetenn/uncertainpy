import subprocess
import datetime
import uncertainpy
import chaospy as cp

from NeuronModelBahl import NeuronModelBahl

parameterlist = [["apical Ra", 261, cp.Uniform(150, 300)],
                 ["soma Ra", 82, cp.Uniform(80, 200)]]#,
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()]]


# model_path = "uncertainpy/models/neuron_models/bahl/"

parameters = uncertainpy.Parameters(parameterlist)
model = NeuronModelBahl(parameters=parameters)

model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                CPUs=1,
                                                save_figures=True,
                                                output_dir_data="../../uncertainpy_results/data/bahl",
                                                output_dir_figures="../../uncertainpy_results/figures/bahl",
                                                nr_mc_samples=10**2,
                                                combined_features=True)


uncertainty.singleParameters()
uncertainty.allParameters()
uncertainty.plot.plotSimulatorResults()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(uncertainty.timePassed())))
