import subprocess
import datetime
import uncertainpy
import chaospy as cp

from lgnSimulator import LgnSimulator
from lgnSimulator_features import LgnSimulatorFeatures

parameterlist = [["w_ri", 0.5, cp.Uniform(-2, 0)],
                 ["w_rc", 2, cp.Uniform(0, 1.0)],
                 ["w_ic", 2, cp.Uniform(0, 4)],
                 ["a_ri", 2, cp.Uniform(0.001, 4)],
                 ["a_rc", 2, cp.Uniform(0.001, 4)],
                 ["a_ic", 2, cp.Uniform(0.001, 4)]]

parameterlist_short = [["w_ic", 0.5, cp.Uniform(0, 4)],
                       ["w_rc", 2, cp.Uniform(0, 1.0)]]

parameters = uncertainpy.Parameters(parameterlist)



config_file_base = "/home/simen/src/lgn-simulator/apps/spatialSummation/spatialSummation.yaml"
config_file = "/home/simen/src/lgn-simulator/apps/spatialSummation/_spatialSummation.yaml"
output_file = "/home/simen/src/lgn-simulator/apps/spatialSummation/spatialSummation.h5"

model = LgnSimulator(parameters=parameters,
                     config_file=config_file,
                     config_file_base=config_file_base,
                     output_file=output_file)

features = LgnSimulatorFeatures()

uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                CPUs=1,
                                                save_figures=True,
                                                features=features,
                                                feature_list="all",
                                                output_dir_data="../../uncertainpy_results/data/milad",
                                                output_dir_figures="../../uncertainpy_results/figures/milad",
                                                rosenblatt=False)


# uncertainty.singleParameters()
uncertainty.allParameters()
uncertainty.plot.plotSimulatorResults()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(uncertainty.timePassed())))
