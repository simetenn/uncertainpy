import uncertainpy as un
import chaospy as cp

from coffee_cup_dependent_function import coffee_cup_dependent

dist_u_env = cp.Uniform(15, 25)
dist_a = cp.Uniform(0.5, 1.5)
dist_kappa = cp.Uniform(-0.075, -0.025)/dist_a


parameterlist = [["kappa", -0.05, dist_a],
                 ["u_env", 20, dist_u_env],
                 ["a", 1, dist_a]]

parameters = un.Parameters(parameterlist)

model = un.Model(coffee_cup_dependent, labels=["time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None)

uncertainty.uncertainty_quantification(rosenblatt=True,
                                       filename="coffee_cup_dependent_rosenblatt",
                                       output_dir_figures="figures_rosenblatt")

uncertainty.uncertainty_quantification(rosenblatt=True)
