import uncertainpy as un
import chaospy as cp

from brunel_network_function import brunel_network


model = un.NestModel(run_function=brunel_network,
                     adaptive=False)

features = un.NetworkFeatures(features_to_run="all")



# SR parameter set

parameterlist = [["eta", 2, cp.Uniform(1.5, 3.5)],
                 ["g", 5, cp.Uniform(1, 3)]]

parameters = un.Parameters(parameterlist)

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       output_dir_figures="figures_brunel_function_SR",
                                       CPUs=1)


uncertainty.uncertainty_quantification(plot_condensed=True,
                                       plot_simulator_results=True)




# AI parameter set

parameterlist = [["eta", 2, cp.Uniform(1.5, 3.5)],
                 ["g", 5, cp.Uniform(1, 3)]]

parameters = un.Parameters(parameterlist)

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       output_dir_figures="figures_brunel_function_AI",
                                       CPUs=1)


uncertainty.uncertainty_quantification(plot_condensed=True,
                                       plot_simulator_results=True)