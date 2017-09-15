import uncertainpy as un
import chaospy as cp

from coffee_cup_dependent_function import coffee_cup_dependent

dist_u_env = cp.Uniform(15, 25)
dist_a = cp.Uniform(0.5, 1.5)
dist_kappa = cp.Uniform(-0.075, -0.025)/dist_a


parameterlist = [["kappa", -0.05, dist_a],
                 ["u_env", 20, dist_u_env],
                 ["a", 1, dist_a]]

C = [[.0001, 0, 0.0009],
     [0, 5, 0],
     [0.0009, 0, .01]]

C = [[.00001, 0, 0.0001],
     [0, 3, 0],
     [0.0001, 0, .01]]

mu = [-0.05, 20, 1]
dist = cp.MvNormal(mu, C)

parameterlist = [["kappa", -0.05, dist_a],
                 ["u_env", 20, dist_u_env],
                 ["a", 1, dist_a]]


parameters = un.Parameters(parameterlist)
parameters.distribution = dist

model = un.Model(coffee_cup_dependent, labels=["time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None,
                                       CPUs=1)

uncertainty.uncertainty_quantification(rosenblatt=True,
                                       filename="coffee_cup_dependent_rosenblatt",
                                       output_dir_figures="figures_rosenblatt",
                                       plot_results=True)
