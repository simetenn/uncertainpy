from model import Model

import numpy as np


class CoffeeCup2DModel(Model):
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

        self.t = np.linspace(0, 360, 150)

        self.dt = t[1] - t[0]


        self.x = np.linspace(0, 10, 500)

        self.u0 = np.zeros(len(self.x))
        for i, t in enumerate(self.x):
            if 4 <= t <= 6:
                self.u0[i] = 95
            else:
                self.u0[i] = 20

    def f(self, x, t):
        ""


    def run(self):

        for i in xrange(len(self.t_points) - 1):
             dt = t[i+1] - t[i]
             k1 = dt*f(x[i], t[i])
             k2 = dt*(x[i] + 0.5*k1, t[i] + 0.5*dt)
             k3 = dt*f(x[i] + 0.5*k2, t[i] + 0.5*dt)
             k4 = dt*f(x[i] + k3, t[i+1])
             x[i+1] = x[i] + (k1 + 2.0*(k2 + k3) + k4)/6.0
