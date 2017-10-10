import numpy as np
from scipy.integrate import odeint

def coffee_cup(kappa=-0.05, T_env=20, T_0=95):
    t = np.linspace(0, 200, 150)

    def f(T, t, kappa, T_env):
        return kappa*(T - T_env)

    U = odeint(f, T_0, t, args=(kappa, T_env))[:, 0]

    return t, U
