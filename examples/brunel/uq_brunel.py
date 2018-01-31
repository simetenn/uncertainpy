import uncertainpy as un
import chaospy as cp

from brunel_network import brunel_network


# Create a Nest model from the brunel network function
model = un.NestModel(run_function=brunel_network)

# Initialize network features
features = un.NetworkFeatures()

# Parametes for the synchronous regular (SR) state
parameter_list = [["eta", None, cp.Uniform(1.5, 3.5)],
                  ["g", None, cp.Uniform(1, 3)],
                  ["delay", None, cp.Uniform(1.5, 3)],
                  ["J_E", None, cp.Uniform(0.05, 0.15)]]
parameters_SR = un.Parameters(parameter_list)

# Parameter for the asynchronous irregular (AI) state
parameter_list = [["eta", None, cp.Uniform(1.5, 2.2)],
                  ["g", None, cp.Uniform(5, 8)],
                  ["delay", None, cp.Uniform(1.5, 3)],
                  ["J_E", None, cp.Uniform(0.05, 0.15)]]
parameters_AI = un.Parameters(parameter_list)

# Set up the problem
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters_SR,
                                  features=features)

# Perform uncertainty quantification
# and save the data and plots under their own name
UQ.quantify(figure_folder="figures_brunel_SR",
            filename="brunel_SR")

# Change the set of parameters
UQ.parameters = parameters_AI

# Perform uncertainty quantification on the new parameter set
# and save the data and plots under their own name
UQ.quantify(figure_folder="figures_brunel_AI",
            filename="brunel_AI")