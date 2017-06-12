import uncertainpy as un

from coffee_cup_class import CoffeeCup

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.Distribution(0.5).uniform)


model = CoffeeCup()

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None)

uncertainty.uncertainty_quantification(plot_condensed=False)
