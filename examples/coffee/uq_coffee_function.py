import uncertainpy as un
import chaospy as cp
import numpy as np
from scipy.integrate import odeint


# Create the coffee cup run function
def coffee_cup(kappa, T_env):
    # Initial temperature and time
    time = np.linspace(0, 200, 150)
    T_0 = 95

    # The equation describing the model
    def f(T, time, kappa, T_env):
        return -kappa*(T - T_env)

    # Solving the equation by integration.
    temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

    # Return time and model results
    return time, temperature


# Create a model from coffee_cup function and add labels
model = un.Model(run_function=coffee_cup,
                 labels=["Time [s]", "Temperature [C]"])

# Create the distributions
kappa_dist = cp.Uniform(0.025, 0.075)
T_env_dist = cp.Uniform(15, 25)

# Define a parameter list and use it to create the Parameters
parameter_list = [["kappa", None, kappa_dist],
                  ["T_env", None, T_env_dist]]
parameters = un.Parameters(parameter_list)

# Perform the uncertainty quantification
uncertainty = un.UncertaintyQuantification(model=model,
                                           parameters=parameters)
# uncertainty.quantify(rosenblatt=True, pc_method="spectral")
uncertainty.monte_carlo(nr_samples=10**2)