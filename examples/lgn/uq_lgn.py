import uncertainpy as un


# Define a parameter list
parameter_list = [["Epas", -67],
                  ["Rm", 22000],
                  ["gna", 0.09],
                  ["nash", -52.6],
                  ["gkdr", 0.37],
                  ["kdrsh", -51.2],
                  ["gcat", 1.17e-5],
                  ["gcal", 0.0009],
                  ["ghbar", 0.00011],
                  ["gahp", 6.4e-5],
                  ["gcanbar", 2e-8]]

# Create the parameters
parameters = un.Parameters(parameter_list)

# Set all parameters to have a uniform distribution
# within a 5% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.05))

# Initialize the model with the start and end time of the stimulus
model = un.NeuronModel(path="dLGN_modelDB/", adaptive=True,
                       stimulus_start=1000, stimulus_end=1900)

# Initialize the features
features = un.SpikingFeatures(features_to_run="all")

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features,
                                  CPUs=7)
UQ.quantify()
