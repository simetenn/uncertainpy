import uncertainpy
from CoffeeCupPointModel import CoffeeCupPointModel

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
parameters.setAllDistributions(uncertainpy.Distribution(0.5).uniform)

model = CoffeeCupPointModel(parameters)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                features=None,
                                                save_figures=True)

uncertainty.PC()
