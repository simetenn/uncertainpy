import uncertainpy as un
import chaospy as cp

from coffee_cup_function import coffee_cup

# Create a model from coffee_cup function and add labels
model = un.Model(run_function=coffee_cup,
                 labels=["Time [s]", "Temperature [C]"])

# Create the distributions
kappa_dist = cp.Uniform(-0.075, -0.025)
T_env_dist = cp.Uniform(15, 25)

# Define a parameter list and use it to create the Parameters
parameter_list = [["kappa", None, kappa_dist],
                  ["T_env", None, T_env_dist]]
parameters = un.Parameters(parameter_list)

# Perform the uncertainty quantification
uncertainty = un.UncertaintyQuantification(model=model,
                                           parameters=parameters)
uncertainty.quantify()
