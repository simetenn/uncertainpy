import uncertainpy as un
import chaospy as cp

from valderrama import valderrama

# Initialize the model
model = un.Model(run=valderrama,
                 labels=["Time (ms)", "Membrane potential (mV)"])

# Define a parameter list
parameters = {"V_0": -10,
              "C_m": 1,
              "gbar_Na": 120,
              "gbar_K": 36,
              "gbar_l": 0.3,
              "m_0": 0.0011,
              "n_0": 0.0003,
              "h_0": 0.9998,
              "E_Na": 112,
              "E_K": -12,
              "E_l": 10.613}

# Create the parameters
parameters = un.Parameters(parameters)

# Set all parameters to have a uniform distribution
# within a 20% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.2))

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters)
data = UQ.quantify()
