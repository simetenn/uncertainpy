import uncertainpy as un
import chaospy as cp

from coffee_cup_function import coffee_cup

kappa_dist = cp.Uniform(-0.075, -0.025)
T_env_dist = cp.Uniform(15, 25)

parameter_list = [["kappa", -0.05, kappa_dist],
                 ["T_env", 20, T_env_dist]]

parameters = un.Parameters(parameter_list)

model = un.Model(coffee_cup, labels=["Time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None,)

uncertainty.uncertainty_quantification(plot_condensed=False)
