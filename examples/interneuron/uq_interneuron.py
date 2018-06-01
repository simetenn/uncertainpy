import uncertainpy as un

# Define a parameter list
parameters= {"gna": 0.09,
             "gkdr": 0.37,
             "gcat": 1.17e-5,
             "gcal": 0.0009,
             "ghbar": 0.00011,
             "gahp": 6.4e-5,
             "gcanbar": 2e-8}

# Create the parameters
parameters = un.Parameters(parameters)

# Set all parameters to have a uniform distribution
# within a 20% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.2))

# Initialize the features
features = un.SpikingFeatures(features_to_run="all")

# Initialize the model with the start and end time of the stimulus
model = un.NeuronModel(path="interneuron_model/", interpolate=True,
                       stimulus_start=1000, stimulus_end=1900)

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features)
# We set the seed to easier be able to reproduce the result
data = UQ.quantify(seed=10)
