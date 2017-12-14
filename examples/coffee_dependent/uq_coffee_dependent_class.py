import uncertainpy as un

from coffee_cup_dependent_class import CoffeeCupDependent

parameter_list = [["kappa", -0.05, None],
                 ["u_env", 20, None],
                 ["alpha", 1, None]]

parameters = un.Parameters(parameter_list)
parameters.set_all_distributions(un.uniform(0.5))

model = CoffeeCupDependent()

uncertainty = un.UncertaintyQuantification(model=model,
                                       parameters=parameters,
                                       features=None)

uncertainty.uncertainty_quantification(plot_condensed=False, rosenblatt=True)
