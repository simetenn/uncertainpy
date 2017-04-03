import uncertainpy as un

from coffe_cup_class import CoffeeCup
from coffee_cup_function import coffe_cup

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]

parameters = un.Parameters(parameterlist)
parameters.setAllDistributions(un.Distribution(0.5).uniform)


# model = CoffeeCup()
model = coffe_cup

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None)

uncertainty.UQ(plot_condensed=False)
