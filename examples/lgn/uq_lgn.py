import uncertainpy as un

path = "dLGN_modelDB/"

# Define a parameter list
parameter_list = [["cap", 1.1, None],
                  ["Rm", 22000, None],
                  ["Vrest", -63, None],
                  ["Epas", -67, None],
                  ["gna", 0.09, None],
                  ["nash", -52.6, None],
                  ["gkdr", 0.37, None],
                  ["kdrsh", -51.2, None],
                  ["gahp", 6.4e-5, None],
                  ["gcat", 1.17e-5, None]]

# Create the parameters
parameters = un.Parameters(parameter_list)

# Set all parameters to have a uniform distribution
# within a 5% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.05))

# Initialize the model with the start and end time of the stimulus
model = un.NeuronModel(path=path, adaptive=True,
                       stimulus_start=1000, stimulus_end=1900)

# Initialize the features
features = un.SpikingFeatures(features_to_run="all")

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  CPUs=7)
UQ.quantify(allow_incomplete=True)
