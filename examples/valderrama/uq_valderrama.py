import uncertainpy as un
import chaospy as cp

from valderrama import Valderrama


parameters = [["V_0", -10, None],
              ["C_m", 1, cp.Uniform(0.8, 1.5)],
              ["gbar_Na", 120, cp.Uniform(64, 260)],
              ["gbar_K", 36, cp.Uniform(26, 49)],
              ["gbar_l", 0.3, cp.Uniform(0.13, 0.5)],
              ["m_0", 0.0011, None],
              ["n_0", 0.0003, None],
              ["h_0", 0.9998, None],
              ["E_Na", 112, cp.Uniform(30, 54)],
              ["E_K", -12, cp.Uniform(-74, -79)],
              ["E_l", 10.613, cp.Uniform(-61, -43)]]


parameters = un.Parameters(parameters)
parameters.set_all_distributions(un.uniform(0.2))

model = Valderrama()

features = un.SpikingFeatures(thresh="auto")
# features = un.EfelFeatures()
exploration = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       allow_incomplete=False)

exploration.uncertainty_quantification(plot_condensed=True,
                                       plot_results=False)
