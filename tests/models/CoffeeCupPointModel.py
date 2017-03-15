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

        self.kappa = -0.05
        self.u_env = 20

        self.u0 = 95
        self.t_points = np.linspace(0, 200, 150)

        self.xlabel = "time [s]"
        self.ylabel = "Temperature [C]"

    def f(self, u, t):
        return self.kappa*(u - self.u_env)

    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        solver = odespy.RK4(self.f)

        solver.set_initial_condition(self.u0)

        self.U, self.t = solver.solve(self.t_points)
        # self.U = np.array([self.U, self.U, self.U])

        return self.t, self.U
