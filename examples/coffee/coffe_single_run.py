import uncertainpy
from CoffeeCupPointModel import CoffeeCupPointModel

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = CoffeeCupPointModel(parameters)

model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                features=None,
                                                CPUs=1,
                                                save_figures=True)

uncertainty.allParameters()
