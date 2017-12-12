from uncertainpy import Model
from scipy.integrate import odeint
import numpy as np

class CoffeeCup(Model):
    """
    """
    def __init__(self):
        Model.__init__(self,
                       labels=["Time [s]", "Temperature [C]"])


    def run(self, kappa=-0.05, T_env=20):
        # Initial temperature and time
        T_0 = 95
        time = np.linspace(0, 200, 100)

        # The equation describing the model
        def f(T, time, kappa, T_env):
            return kappa*(T - T_env)

        # Solving the equation by integration.
        temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

        return time, temperature
