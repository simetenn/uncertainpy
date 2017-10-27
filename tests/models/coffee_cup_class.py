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
        T_0 = 95
        t = np.linspace(0, 200, 100)

        def f(T, t, kappa, T_env):
            return kappa*(T - T_env)

        U = odeint(f, T_0, t, args=(kappa, T_env))[:, 0]

        return t, U
