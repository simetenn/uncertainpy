import numpy as np
from scipy.integrate import odeint

def coffee_cup(kappa, T_env):
    time = np.linspace(0, 200, 150)
    T_0 = 95

    # The equation describing the model
    def f(T, time, kappa, T_env):
        return kappa*(T - T_env)

    # Solving the equation by integration.
    temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

    return time, temperature
