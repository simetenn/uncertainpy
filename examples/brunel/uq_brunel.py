import uncertainpy as un
import chaospy as cp

from brunel import brunel_network

# Create a Nest model from the brunel network function
# We set ``ignore=True`` since we are not interested in
# the model result itself.
# This is recommended for NEST models as long as you do not
# need the model results, since the uncertainty calculations for the
# for the model results require much time and memory.
model = un.NestModel(run=brunel_network, ignore=True)

# Parametes for the synchronous regular (SR) state
parameters = {"eta": cp.Uniform(1.5, 3.5),
              "g": cp.Uniform(1, 3),
              "delay": cp.Uniform(1.5, 3),
              "J_E": cp.Uniform(0.05, 0.15)}
parameters_SR = un.Parameters(parameters)

# Parameter for the asynchronous irregular (AI) state
parameters = {"eta": cp.Uniform(1.5, 2.2),
              "g": cp.Uniform(5, 8),
              "delay": cp.Uniform(1.5, 3),
              "J_E": cp.Uniform(0.05, 0.15)}
parameters_AI = un.Parameters(parameters)

# Initialize network features
features = un.NetworkFeatures()

# Set up the problem
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters_SR,
                                  features=features)

# Perform uncertainty quantification
# and save the data and plots under their own name
# We set the seed to easier be able to reproduce the result
UQ.quantify(figure_folder="figures_brunel_SR",
            filename="brunel_SR",
            seed=10)

# Change the set of parameters
UQ.parameters = parameters_AI

# Perform uncertainty quantification on the new parameter set
# and save the data and plots under their own name
# We set the seed to easier be able to reproduce the result
data = UQ.quantify(figure_folder="figures_brunel_AI",
                   filename="brunel_AI",
                   seed=10)