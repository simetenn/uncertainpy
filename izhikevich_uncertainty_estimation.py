import subprocess
import datetime
import uncertainpy

memory = uncertainpy.utils.Memory(10)
memory.start()

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.models.IzhikevichModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


exploration = uncertainpy.UncertaintyEstimations(model,
                                                 CPUs=8,
                                                 save_figures=True,
                                                 feature_list="all",
                                                 output_dir_data="data/izhikevich",
                                                 output_dir_figures="figures/izhikevich",
                                                 nr_mc_samples=10**2)

percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
# percentages = np.linspace(0.01, 0.25, 50)
# percentages = [0.01, 0.03, 0.05, 0.07, 0.09]

test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

# mc_samples = [50, 100, 200, 500, 1000, 1500, 2000]
# exploration.compareMC(mc_samples)

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
