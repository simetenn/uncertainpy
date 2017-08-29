import uncertainpy as un

from coffee_cup_function import coffee_cup

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.uniform(0.5))


model = un.Model(coffee_cup, labels=["time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None)

uncertainty.uncertainty_quantification(plot_condensed=False)
