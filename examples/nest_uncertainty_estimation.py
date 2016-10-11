import subprocess
import datetime
import uncertainpy

memory = uncertainpy.utils.Memory(10)
memory.start()

parameterlist = [["J_E", 4, None],
                 ["g", 4, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.models.NestNetwork(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.99).uniform)


exploration = uncertainpy.UncertaintyEstimations(model,
                                                 CPUs=8,
                                                 save_figures=True,
                                                 feature_list="all",
                                                 output_dir_data="data/nest",
                                                 output_dir_figures="figures/nest",
                                                 nr_mc_samples=10**2)

#percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
# percentages = np.linspace(0.01, 0.25, 50)
percentages = [0.01, 0.03, 0.05, 0.07, 0.09]

test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

# mc_samples = [50, 100, 200, 500, 1000, 1500, 2000]
# exploration.compareMC(mc_samples)

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
