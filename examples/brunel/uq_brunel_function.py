import uncertainpy as un

from brunel_network_function import brunel_network

parameterlist = [["eta", 2, None],
                 ["g", 5, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.uniform(0.1))

def function(t, U):
    return t, U


model = un.NestModel(run_function=brunel_network,
                     adaptive=False)


print model.name

# features = un.NetworkFeatures(features_to_run="all")


# uncertainty = un.UncertaintyEstimation(model,
#                                        parameters=parameters,
#                                        features=features,
#                                        output_dir_figures="figures_brunel_function",
#                                        CPUs=1)


# uncertainty.uncertainty_quantification(plot_condensed=True,
#                                        plot_simulator_results=True)
