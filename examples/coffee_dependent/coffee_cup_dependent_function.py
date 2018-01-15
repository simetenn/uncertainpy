import numpy as np
from scipy.integrate import odeint

def coffee_cup_dependent(kappa_hat, T_env, alpha):
    # Initial temperature and time
    time = np.linspace(0, 200, 150)
    T_0 = 95

    # The equation describing the model
    def f(T, time, alpha, kappa_hat, T_env):
        return -alpha*kappa_hat*(T - T_env)

    # Solving the equation by integration.
    values = odeint(f, T_0, time, args=(alpha, kappa_hat, T_env))[:, 0]

    # Return time and model results
    return time, values
