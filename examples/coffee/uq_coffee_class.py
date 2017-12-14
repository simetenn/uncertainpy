import uncertainpy as un
import chaospy as cp

from coffee_cup_class import CoffeeCup

T_env_dist = cp.Uniform(15, 25)
kappa_dist = cp.Uniform(-0.075, -0.025)

parameter_list = [["kappa", -0.05, kappa_dist],
                  ["T_env", 20, T_env_dist]]

parameters = un.Parameters(parameter_list)
model = CoffeeCup()

uncertainty = un.UncertaintyQuantification(model=model,
                                       parameters=parameters,
                                       features=None)

uncertainty.quantify(plot_condensed=False)
