import uncertainpy as un
import chaospy as cp

from valderrama import Valderrama


parameters = [["V_rest", -10, None],
              ["Cm", 1, cp.Uniform(0.8, 1.5)],
              ["gbar_Na", 120, cp.Uniform(64, 260)],
              ["gbar_K", 36, cp.Uniform(26, 49)],
              ["gbar_l", 0.3, cp.Uniform(0.13, 0.5)],
              ["m0", 0.0011, None],
              ["n0", 0.0003, None],
              ["h0", 0.9998, None],
              ["E_Na", 112, cp.Uniform(30, 54)],
              ["E_K", -12, cp.Uniform(-74, -79)],
              ["E_l", 10.613, cp.Uniform(-61, -43)]]


parameters = un.Parameters(parameters)
parameters.setAllDistributions(un.Distribution(0.2).uniform)

model = Valderrama()

features = un.NeuronFeatures(thresh="auto")
exploration = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features)

exploration.UQ(plot_condensed=False, plot_simulator_results=True)
