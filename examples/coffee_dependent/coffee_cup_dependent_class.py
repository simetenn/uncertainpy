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
        t = np.linspace(0, 200, 150)

        def f(u, t, kappa, u_env, alpha):
            return alpha*kappa*(u - u_env)

        U = odeint(f, u0, t, args=(kappa, u_env, alpha))[:, 0]

        return t, U
