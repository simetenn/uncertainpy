import uncertainpy as un
import chaospy as cp                       # To create distributions
import numpy as np                         # For the time array
from scipy.integrate import odeint         # To integrate our equation


# Create the coffee cup model
class CoffeeCup(un.Model):
    # Add labels to the model by calling the constructor of the parent un.Model
    def __init__(self):
        super(CoffeeCup, self).__init__(labels=["Time (s)", "Temperature (C)"])


    # Define the run method
    def run(self, kappa, T_env):
        # Initial temperature and time array
        time = np.linspace(0, 200, 150)            # Minutes
        T_0 = 95                                   # Celsius

        # The equation describing the model
        def f(T, time, kappa, T_env):
            return -kappa*(T - T_env)

        # Solving the equation by integration.
        temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

        # Return time and model output
        return time, temperature


# Initialize the model
model = CoffeeCup()

# Create the distributions
kappa_dist = cp.Uniform(0.025, 0.075)
T_env_dist = cp.Uniform(15, 25)

# Define the parameters dictionary
parameters = {"kappa": kappa_dist, "T_env": T_env_dist}

# and use it to create the parameters
parameters = un.Parameters(parameters)

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model, parameters=parameters)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
# We set the seed to easier be able to reproduce the result
data = UQ.quantify(seed=10)