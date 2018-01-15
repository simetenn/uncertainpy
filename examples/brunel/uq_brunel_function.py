import uncertainpy as un
import chaospy as cp

from brunel_network_function import brunel_network


model = un.NestModel(run_function=brunel_network,
                     adaptive=False)

features = un.NetworkFeatures(features_to_run="all")



# SR parameter set
# Synchronous regular (SR) states, where neurons are
# almost fully synchronized in a few clusters and be-
# have  as  oscillators  when  excitation  dominates  in-
# hibition and synaptic time distributions are sharply
# peaked
parameter_list = [["eta", None, cp.Uniform(1.5, 3.5)],
                 ["g", None, cp.Uniform(1, 3)],
                 ["delay", None, cp.Uniform(1.5, 3)],
                 ["J_E", None, cp.Uniform(0.05, 0.15)]]

parameters = un.Parameters(parameter_list)

UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  figure_folder="figures_brunel_function_SR",
                                  CPUs=7,
                                  allow_incomplete=True)

UQ.quantify(plot_condensed=True,
            plot_model=False,
            filename="brunel_function_SR")




# AI parameter set
# Asynchronous irregular (AI) states, with stationary
# global activity but strongly irregular individual firing
# at low rates when inhibition dominates excitation in
# an intermediate range of external frequencies
parameter_list = [["eta", None, cp.Uniform(1.5, 2.2)],
                 ["g", None, cp.Uniform(5, 8)],
                 ["delay", None, cp.Uniform(1.5, 3)],
                 ["J_E", None, cp.Uniform(0.05, 0.15)]]

parameters = un.Parameters(parameter_list)

UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  figure_folder="figures_brunel_function_AI",
                                  CPUs=7,
                                  allow_incomplete=True)

UQ.quantify(plot_condensed=True,
            plot_model=False,
            filename="brunel_function_AI")


# the border between SR and AI parameter set
parameter_list = [["eta", None, cp.Uniform(1.5, 2.2)],
                 ["g", None, cp.Uniform(1, 8)],
                 ["delay", None, cp.Uniform(1.5, 3)],
                 ["J_E", None, cp.Uniform(0.05, 0.15)]]

parameters = un.Parameters(parameter_list)

UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  figure_folder="figures_brunel_function_border",
                                  CPUs=7,
                                  allow_incomplete=True)

UQ.quantify(plot_condensed=True,
            plot_model=False,
            filename="brunel_function_border")