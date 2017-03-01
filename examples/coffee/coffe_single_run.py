import uncertainpy
from CoffeeCupPointModel import CoffeeCupPointModel

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]


model = CoffeeCupPointModel(parameterlist)
model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model, features=None)

uncertainty.UQ(plot_condensed=False)
