import uncertainpy as un

from brunel_network_class import BrunelNetwork

parameterlist = [["J_E", 4, None],
                 ["g", 4, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.Distribution(0.1).uniform)

model = BrunelNetwork()

features = un.NetworkFeatures(features_to_run="all")

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       CPUs=1)


uncertainty.uncertainty_quantification(plot_condensed=False,
                                       plot_simulator_results=True)
