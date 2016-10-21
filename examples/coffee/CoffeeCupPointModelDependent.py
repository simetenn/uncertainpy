from uncertainpy import Model

import numpy as np
import odespy


class CoffeeCupPointModelDependent(Model):
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

        self.kappa = -0.1
        self.alpha = 0.1
        self.u_env = 20

        self.u0 = 95
        self.t_points = np.linspace(0, 360, 150)

        self.x = np.linspace(0, 10, 500)

    def f(self, u, t):
        return self.kappa*self.alpha*(u - self.u_env)

    def run(self):

        solver = odespy.RK4(self.f)

        solver.set_initial_condition(self.u0)

        self.U, self.t = solver.solve(self.t_points)
        return self.t, self.U
