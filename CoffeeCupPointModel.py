from model import Model

import numpy as np
import pylab as plt
import odespy

# The class name and file name must be the same
class CoffeeCupPointModel(Model):
    """
    The model must be able to handle these calls

    simulation = model()
    simulation.load()
    simulation.setParameters(parameters -> dictionary)
    simulation.run()
    simulation.save(current_process -> int)

    simulation.cmd()
    """
    def __init__(self):
        Model.__init__(self)

        self.kappa = -0.01
        self.u_env = 20

        self.u0 = 95
        self.t_points = np.linspace(0, 360, 150)

        self.x = np.linspace(0, 10, 500)

        # self.u0 = np.zeros(len(self.x))
        # for i, t in enumerate(self.x):
        #     if 4 <= t <= 6:
        #         self.u0[i] = 95

    def f(self, u, t):
        return self.kappa*(u - self.u_env)

    def run(self):

        solver = odespy.RK4(self.f)

        solver.set_initial_condition(self.u0)

        self.U, self.t = solver.solve(self.t_points)
        return self.t, self.U
