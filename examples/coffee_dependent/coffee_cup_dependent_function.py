import numpy as np
from scipy.integrate import odeint

def coffe_cup_dependent(kappa=-0.05, u_env=20, alpha=1):
    u0 = 95
    t = np.linspace(0, 200, 150)

    def f(u, t, kappa, u_env, alpha):
        return alpha*kappa*(u - u_env)

    U = odeint(f, u0, t, args=(kappa, u_env, alpha))[:, 0]

    return t, U
