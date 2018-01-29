import uncertainpy as un
import chaospy as cp

from valderrama import valderrama

# Define a parameter list
parameter_list = [["V_0", -10, None],
                  ["C_m", 1, None],
                  ["gbar_Na", 120, None],
                  ["gbar_K", 36, None],
                  ["gbar_l", 0.3, None],
                  ["m_0", 0.0011, None],
                  ["n_0", 0.0003, None],
                  ["h_0", 0.9998, None],
                  ["E_Na", 112, None],
                  ["E_K", -12, None],
                  ["E_l", 10.613, None]]

# Create the parameters
parameters = un.Parameters(parameter_list)

# Set all parameters to have a uniform distribution
# within a 20% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.2))

# Initialize the model
model = un.Model(run_function=valderrama,
                 labels=["Time (ms)", "Membrane potential (mV)"])

# Initialize features, with automatic detection of spikes
features = un.SpikingFeatures(threshold="auto")

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters,
                                  features=features)
UQ.quantify()
