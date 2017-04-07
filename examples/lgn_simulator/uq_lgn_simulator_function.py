import uncertainpy as un
import chaospy as cp

from lgn_simulator_function import lgn_simulator
from lgn_simulator_features_function import irfmax, irfmin, irf_size


parameterlist = [["w_ri", 0.5, cp.Uniform(-2, 0)],
                 ["w_rc", 2, cp.Uniform(0, 1.0)],
                 ["w_ic", 2, cp.Uniform(0, 4)],
                 ["a_ri", 2, cp.Uniform(0.001, 4)],
                 ["a_rc", 2, cp.Uniform(0.001, 4)],
                 ["a_ic", 2, cp.Uniform(0.001, 4)]]

parameters = un.Parameters(parameterlist)

model = lgn_simulator
features = [irfmax, irfmin, irf_size]

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       CPUs=1)

uncertainty.uncertainty_quantification(uncertain_parameters=["w_rc", "w_ri"])
