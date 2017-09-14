import uncertainpy as un
import chaospy as cp

from coffee_cup_function import coffee_cup

dist_u_env = cp.Uniform(15, 25)
dist_kappa = cp.Uniform(-0.075, -0.025)

parameterlist = [["kappa", -0.05, dist_kappa],
                 ["u_env", 20, dist_u_env]]

parameters = un.Parameters(parameterlist)


model = un.Model(coffee_cup, labels=["time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None,)

uncertainty.uncertainty_quantification(plot_condensed=False)
