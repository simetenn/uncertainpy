import uncertainpy as un
import chaospy as cp

from HodgkinHuxleyModel import HodgkinHuxleyModel


parameterlist = [["V_rest", -65, None],
                 ["Cm", 1, cp.Uniform(0.8, 1.5)],
                 ["gbar_Na", 120, cp.Uniform(80, 160)],
                 ["gbar_K", 36, cp.Uniform(26, 49)],
                 ["gbar_l", 0.3, cp.Uniform(0.13, 0.5)],
                 ["E_Na", 50, cp.Uniform(30, 54)],
                 ["E_K", -77, cp.Uniform(-74, -79)],
                 ["E_l", -50.613, cp.Uniform(-61, -43)]]


parameters = un.Parameters(parameterlist)
parameters.setAllDistributions(un.Distribution(0.5).uniform)


model = HodgkinHuxleyModel()
features = un.NeuronFeatures(features_to_run="all")

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=features)


uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features)
uncertainty.UQ()
