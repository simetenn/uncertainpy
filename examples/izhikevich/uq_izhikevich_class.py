import uncertainpy as un
from izhikevich_class import Izhikevich

# Define a parameter list
parameter_list = [["a", 0.02, None],
                  ["b", 0.2, None],
                  ["c", -65, None],
                  ["d", 8, None]]

# Create the parameters
parameters = un.Parameters(parameter_list)

# Set all parameters to have a uniform distribution
# within a 50% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.5))

# Initialize the model
model = Izhikevich()
features = un.SpikingFeatures(features_to_run="all")

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters,
                                  features=features)
data = UQ.quantify()
