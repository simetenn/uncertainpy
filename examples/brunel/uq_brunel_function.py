import uncertainpy as un
import chaospy as cp

from brunel_network_function import brunel_network


# Create a Nest model from the brunel network function
model = un.NestModel(run_function=brunel_network)

# Initialize network features
features = un.NetworkFeatures()


# Parametes for the Synchronous regular (SR) state
parameter_list = [["eta", None, cp.Uniform(1.5, 3.5)],
                  ["g", None, cp.Uniform(1, 3)],
                  ["delay", None, cp.Uniform(1.5, 3)],
                  ["J_E", None, cp.Uniform(0.05, 0.15)]]

parameters = un.Parameters(parameter_list)

# Perform uncertainty quantification
# using 7 CPUs,
# allowing incomplete features to be used and saving
# the data and plots under their own name
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  CPUs=7)

UQ.quantify(figure_folder="figures_brunel_function_SR",
            filename="brunel_function_SR")




# Parameter for the asynchronous irregular (AI) state
parameter_list = [["eta", None, cp.Uniform(1.5, 2.2)],
                  ["g", None, cp.Uniform(5, 8)],
                  ["delay", None, cp.Uniform(1.5, 3)],
                  ["J_E", None, cp.Uniform(0.05, 0.15)]]

parameters = un.Parameters(parameter_list)

# Perform uncertainty quantification
# using 7 CPUs,
# allowing incomplete features to be used and saving
# the data and plots under their own name
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  CPUs=7)

UQ.quantify(figure_folder="figures_brunel_function_AI",
            filename="brunel_function_AI")