from uncertainpy import Model
from scipy.integrate import odeint
import numpy as np

# Create the coffee cup model
class CoffeeCup(Model):
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