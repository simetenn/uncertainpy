from uncertainpy import Model

import numpy as np
import odespy


class CoffeeCupPointModel(Model):
    """
    """
    def __init__(self):
        Model.__init__(self,
                       xlabel="time [s]",
                       ylabel="Temperature [C]")



    def run(self, kappa=-0.05, u_env=20):
        u0 = 95
        t_points = np.linspace(0, 200, 150)

        def f(u, t):
            return kappa*(u - u_env)


        solver = odespy.RK4(f)
        solver.set_initial_condition(u0)

        U, t = solver.solve(t_points)

        return t, U
