import uncertainpy as un
import chaospy as cp

from coffee_cup_function import coffee_cup

T_0_dist = cp.Uniform(90, 100)
T_env_dist = cp.Uniform(15, 25)
kappa_dist = cp.Uniform(-0.075, -0.025)

parameterlist = [["kappa", -0.05, kappa_dist],
                 ["T_env", 20, T_env_dist],
                 ["T_0", 95, None]]

parameters = un.Parameters(parameterlist)

model = un.Model(coffee_cup, labels=["Time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None,)

uncertainty.uncertainty_quantification(plot_condensed=False)
