from uncertainpy import Model
import numpy as np
from scipy.integrate import odeint


class CoffeeCupDependent(Model):
    """
    """
    def __init__(self):
        Model.__init__(self,
                       labels=["time [s]", "Temperature [C]"])


    def run(self, kappa=-0.05, u_env=20, alpha=1):
        u0 = 95
        time = np.linspace(0, 200, 150)

        def f(u, time, kappa, u_env, alpha):
            return alpha*kappa*(u - u_env)

        values = odeint(f, u0, time, args=(kappa, u_env, alpha))[:, 0]

        return time, values
