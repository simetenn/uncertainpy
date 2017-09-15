import numpy as np
from scipy.integrate import odeint

def coffee_cup_dependent(kappa=-0.05, u_env=20, a=1):
    u0 = 95
    t = np.linspace(0, 200, 150)

    print "kappa ", kappa
    print "a ", a
    print "u_env", u_env

    def f(u, t, kappa, u_env, a):
        return a*kappa*(u - u_env)

    U = odeint(f, u0, t, args=(kappa, u_env, a))[:, 0]

    return t, U
