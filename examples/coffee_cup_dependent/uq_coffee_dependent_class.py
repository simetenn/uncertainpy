import uncertainpy as un
import numpy as np
from scipy.integrate import odeint
import chaospy as cp


# Create the coffee cup model
class CoffeeCupDependent(un.Model):
    def __init__(self):
        # Add labels to the model by calling the constructor of the parent un.Model
        super(CoffeeCupDependent, self).__init__(labels=["Time (s)", "Temperature (C)"])


    # Define the run function
    def run(self, kappa_hat, T_env, alpha):
        # Initial temperature and time
        time = np.linspace(0, 200, 150)            # Minutes
        T_0 = 95                                   # Celsius

        # The equation describing the model
        def f(T, time, alpha, kappa_hat, T_env):
            return -alpha*kappa_hat*(T - T_env)

        # Solving the equation by integration.
        values = odeint(f, T_0, time, args=(alpha, kappa_hat, T_env))[:, 0]

        # Return time and model results
        return time, values


# Initialize the model
model = CoffeeCupDependent()

# Create the distributions
T_env_dist = cp.Uniform(15, 25)
alpha_dist = cp.Uniform(0.5, 1.5)
kappa_hat_dist = cp.Uniform(0.025, 0.075)/alpha_dist

# Define the parameters dictionary
parameters = {"alpha": alpha_dist,
              "kappa_hat": kappa_hat_dist,
              "T_env": T_env_dist}

# We can use the parameters dictionary directly
# when we set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model, parameters=parameters)

# Perform the uncertainty quantification,
# which automatically use the Rosenblatt transformation
# We set the seed to easier be able to reproduce the result
data = UQ.quantify(seed=10)
