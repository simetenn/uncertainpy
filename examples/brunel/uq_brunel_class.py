import uncertainpy as un

from brunel_network_class import BrunelNetwork

parameter_list = [["J_E", 4, None],
                 ["g", 4, None]]

parameters = un.Parameters(parameter_list)
parameters.set_all_distributions(un.uniform(0.1))

model = BrunelNetwork()

features = un.NetworkFeatures(features_to_run="all")

UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  figure_folder="figures_brunel_class",
                                  CPUs=1)


UQ.quantify(plot_condensed=False,
            plot_model=True)
