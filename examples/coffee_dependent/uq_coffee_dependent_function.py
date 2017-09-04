import uncertainpy as un

from coffee_cup_dependent_function import coffee_cup_dependent

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None],
                 ["alpha", 1, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.uniform(0.5))

model = un.Model(coffee_cup_dependent, labels=["time [s]", "Temperature [C]"])

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=None)

uncertainty.uncertainty_quantification(rosenblatt=True,
                                       filename="coffee_cup_dependent_rosenblatt",
                                       output_dir_figures="figures_rosenblatt")

uncertainty.uncertainty_quantification(rosenblatt=False)
