import uncertainpy as un

from brunel_network_class import BrunelNetwork

parameterlist = [["J_E", 4, None],
                 ["g", 4, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.uniform(0.1))

model = BrunelNetwork()

features = un.NetworkFeatures(features_to_run="all")

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       output_dir_figures="figures_brunel_class",
                                       CPUs=1)


uncertainty.uncertainty_quantification(plot_condensed=False,
                                       plot_model_results=True)
