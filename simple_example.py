import uncertainpy
import numpy as np
import odespy

class YourOwnCoffeeModel(uncertainpy.Model):
    def __init__(self, parameters=None):
        uncertainpy.Model.__init__(self, parameters=parameters)

        self.kappa = -0.01
        self.u_env = 20

        self.u0 = 95
        self.t_points = np.linspace(0, 360, 150)

    def run(self):
        def f(u, t):
            return self.kappa*(u - self.u_env)

        solver = odespy.RK4(f)
        solver.set_initial_condition(self.u0)

        self.U, self.t = solver.solve(self.t_points)

        return self.t, self.U

class CoffeeFeatures(uncertainpy.GeneralFeatures):
    def averageValue(self):
        return np.mean(self.U)


if __name__ == "__main__":
    parameterlist = [["kappa", -0.01, None],
                     ["u_env", 20, None]]

    parameters = uncertainpy.Parameters(parameterlist)
    model = YourOwnCoffeeModel(parameters)
    distributions = {"uniform": [0.01, 0.02]}

    exploration = uncertainpy.UncertaintyEstimations(model,
                                                     distributions,
                                                     features=CoffeeFeatures(),
                                                     output_dir_data="data/coffee",
                                                     save_figures=True)
