import uncertainpy as un

from brunel_network_class import BrunelNetwork
from brunel_network_function import brunel_network

parameterlist = [["J_E", 4, None],
                 ["g", 4, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.Distribution(0.5).uniform)

# model = BrunelNetwork()
model = brunel_network

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=None)

uncertainty.uncertainty_quantification(plot_condensed=False,
                                       plot_simulator_results=True)
