from uncertainpy import Model

import numpy as np
import odespy


class CoffeeCupPointModel(Model):
    """
    The model must be able to handle these calls

    simulation = model()
    simulation.load()
    simulation.setParameterValues(parameters -> dictionary)
    simulation.run()
    simulation.save(current_process -> int)

    simulation.cmd()
    """
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.xlabel = "time [s]"
        self.ylabel = "Temperature [C]"

    def f(self, u, t):
        return self.kappa*(u - self.u_env)

    def run(self, kappa=-0.05, u_env=20):
        u0 = 95
        t_points = np.linspace(0, 200, 150)

        solver = odespy.RK4(self.f)

        solver.set_initial_condition(u0)

        U, t = solver.solve(t_points)


        return t, U
