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
        Model.__init__(self, parameters=parameters, xlabel="time [s]", ylabel="Temperature [C]")

        self.kappa = -0.05
        self.u_env = 20

        self.u0 = 95
        self.t_points = np.linspace(0, 200, 150)


    def run(self):

        def f(self, u, t):
            return self.kappa*(u - self.u_env)


        solver = odespy.RK4(f)

        solver.set_initial_condition(self.u0)

        self.U, self.t = solver.solve(self.t_points)
