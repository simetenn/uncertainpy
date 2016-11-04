import uncertainpy
import chaospy as cp

from ValderramaHodgkinHuxleyModel import ValderramaHodgkinHuxleyModel

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


parameters = uncertainpy.Parameters(parameters)

model = ValderramaHodgkinHuxleyModel(parameters=parameters)
model.setAllDistributions(uncertainpy.Distribution(0.2).uniform)

features = uncertainpy.NeuronFeatures(features_to_run="all", thresh="auto")

exploration = uncertainpy.UncertaintyEstimation(model,
                                                seed=10,
                                                features=features,
                                                CPUs=7,
                                                save_figures=True,
                                                rosenblatt=False,
                                                figureformat=".pdf")

exploration.allParameters()
exploration.plot.plotSimulatorResults()
