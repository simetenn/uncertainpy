import numpy as np
from scipy.integrate import odeint

def coffee_cup_dependent(kappa=-0.05, u_env=20, a=1):
    u0 = 95
    t = np.linspace(0, 200, 150)

    def f(u, t, a, u_env, b):
        return a*kappa*(u - u_env)

    U = odeint(f, u0, t, args=(kappa, u_env, a))[:, 0]

    return t, U
