import uncertainpy as un
import chaospy as cp

from coffee_cup_dependent_function import coffee_cup_dependent

dist_u_env = cp.Uniform(15, 25)
dist_a = cp.Uniform(0.5, 1.5)
dist_kappa = cp.Uniform(-0.075, -0.025)/dist_a

parameterlist = [["kappa", -0.05, dist_kappa],
                 ["u_env", 20, dist_u_env],
                 ["a", 1, dist_a]]

parameters = un.Parameters(parameterlist)

model = un.Model(coffee_cup_dependent, labels=["time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters)

uncertainty.uncertainty_quantification(rosenblatt=True,
                                       output_dir_figures="figures_coffee_dependent",
                                       filename="coffee_dependent")

C = [[.0001, 0, 0.0005],
     [0, 5, 0],
     [0.0005, 0, .01]]
mu = [-0.05, 20, 1]
dist = cp.MvNormal(mu, C)


uncertainty.parameters.distribution = dist

uncertainty.uncertainty_quantification(rosenblatt=True,
                                       output_dir_figures="figures_coffee_dependent_small_correlation",
                                       filename="coffee_dependent_small_correlation")

C = [[.0001, 0, 0.00099999],
     [0, 5, 0],
     [0.00099999, 0, .01]]
mu = [-0.05, 20, 1]
dist = cp.MvNormal(mu, C)


uncertainty.parameters.distribution = dist
uncertainty.uncertainty_quantification(rosenblatt=True,
                                       output_dir_figures="figures_coffee_dependent_large_correlation",
                                       filename="coffee_dependent_large_correlation")


C = [[.0001, 0, -0.00099999],
     [0, 5, 0],
     [-0.00099999, 0, .01]]
mu = [-0.05, 20, 1]
dist = cp.MvNormal(mu, C)


uncertainty.parameters.distribution = dist
uncertainty.uncertainty_quantification(rosenblatt=True,
                                       output_dir_figures="figures_coffee_dependent_large_reverse_correlation",
                                       filename="coffee_dependent_large_reverse_correlation")

dist_u_env = cp.Uniform(15, 25)
dist_a = cp.Uniform(0.5, 1.5)
dist_kappa = cp.Uniform(-0.075, -0.025)

parameterlist = [["kappa", -0.05, dist_kappa],
                 ["u_env", 20, dist_u_env],
                 ["a", 1, dist_a]]

parameters = un.Parameters(parameterlist)

uncertainty.parameters = parameters
uncertainty.uncertainty_quantification(rosenblatt=True,
                                       output_dir_figures="figures_coffee_dependent_no_correlation_rosenblatt",
                                       filename="coffee_dependent_no_correlation_rosenblatt")

uncertainty.uncertainty_quantification(rosenblatt=False,
                                       output_dir_figures="figures_coffee_dependent_no_correlation",
                                       filename="coffee_dependent_no_correlation")

parameterlist = [["a", 1, None],
                 ["u_env", 20, None],
                 ["kappa", -0.05, None]]

# C = [[3, 0, 0],
#      [0, .0001, 0.0005],
#      [0, 0.0005, .01]]
C = [[.01, 0, 0.0005],
     [0, 5, 0],
     [0.0005, 0, .0001]]
mu = [1, 20, -0.05]
dist = cp.MvNormal(mu, C)

parameters = un.Parameters(parameterlist)
parameters.distribution = dist

uncertainty.parameters = parameters


uncertainty.uncertainty_quantification(rosenblatt=True,
                                       output_dir_figures="figures_coffee_dependent_reverse_order",
                                       filename="coffee_dependent_reverse_order")
