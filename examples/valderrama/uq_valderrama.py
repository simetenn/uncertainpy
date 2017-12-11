import uncertainpy as un
import chaospy as cp

from valderrama import Valderrama

parameter_list = [["V_0", -10, None],
                  ["C_m", 1, None],
                  ["gbar_Na", 120, None],
                  ["gbar_K", 36, None],
                  ["gbar_l", 0.3, None],
                  ["m_0", 0.0011, None],
                  ["n_0", 0.0003, None],
                  ["h_0", 0.9998, None],
                  ["E_Na", 112, None],
                  ["E_K", -12, None],
                  ["E_l", 10.613, None]]

parameters = un.Parameters(parameter_list)
parameters.set_all_distributions(un.uniform(0.2))

model = Valderrama()

features = un.SpikingFeatures(threshold="auto")
# features = un.EfelFeatures()
exploration = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       allow_incomplete=True)

exploration.uncertainty_quantification(plot_condensed=True,
                                       plot_results=False)
