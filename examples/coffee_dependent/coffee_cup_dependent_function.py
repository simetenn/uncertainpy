import numpy as np
from scipy.integrate import odeint

def coffee_cup_dependent(beta=-0.05, T_env=20, alpha=1):
    T_0 = 95
    t = np.linspace(0, 200, 150)

    def f(T, t, alpha, beta, T_env):
        return alpha*beta*(T - T_env)

    U = odeint(f, T_0, t, args=(alpha, beta, T_env))[:, 0]

    return t, U
