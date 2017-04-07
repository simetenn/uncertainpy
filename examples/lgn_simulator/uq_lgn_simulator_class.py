import uncertainpy as un
import chaospy as cp

from lgn_simulator_class import LgnSimulator
from lgn_simulator_features_class import LgnSimulatorFeatures


parameterlist = [["w_ri", 0.5, cp.Uniform(-2, 0)],
                 ["w_rc", 2, cp.Uniform(0, 1.0)],
                 ["w_ic", 2, cp.Uniform(0, 4)],
                 ["a_ri", 2, cp.Uniform(0.001, 4)],
                 ["a_rc", 2, cp.Uniform(0.001, 4)],
                 ["a_ic", 2, cp.Uniform(0.001, 4)]]

parameterlist_short = [["w_ri", 0.5, cp.Uniform(-2, 0)],
                       ["w_rc", 2, cp.Uniform(0, 1.0)]]

parameters = un.Parameters(parameterlist_short)


config_file_base = "/home/simen/src/lgn-simulator/apps/spatialSummation/spatialSummation.yaml"
config_file = "/home/simen/src/lgn-simulator/apps/spatialSummation/_spatialSummation.yaml"
output_file = "/home/simen/src/lgn-simulator/apps/spatialSummation/spatialSummation.h5"

model = LgnSimulator(config_file=config_file,
                     config_file_base=config_file_base,
                     output_file=output_file)

features = LgnSimulatorFeatures()

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       CPUs=1)

uncertainty.uncertainty_quantification(uncertain_parameters=["w_rc", "w_ri"])
