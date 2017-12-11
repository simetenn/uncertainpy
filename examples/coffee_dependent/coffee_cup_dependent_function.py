import numpy as np
from scipy.integrate import odeint

def coffee_cup_dependent(kappa_hat=-0.05, T_env=20, alpha=1):
    T_0 = 95
    time = np.linspace(0, 200, 150)

    def f(T, time, alpha, kappa_hat, T_env):
        return alpha*kappa_hat*(T - T_env)

    values = odeint(f, T_0, time, args=(alpha, kappa_hat, T_env))[:, 0]

    return time, values
