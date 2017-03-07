import uncertainpy as un
from CoffeeCupPointModelDependent import CoffeeCupPointModelDependent

parameterlist = [["kappa", -0.1, None],
                 ["gamma", 0.1, None],
                 ["u_env", 20, None]]


model = CoffeeCupPointModelDependent(parameterlist)
model.setAllDistributions(un.Distribution(0.5).uniform)

uncertainty = un.UncertaintyEstimation(model, save_figures=True)

uncertainty.UQ(rosenblatt=True)
