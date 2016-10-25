import uncertainpy
from CoffeeCupPointModelDependent import CoffeeCupPointModelDependent

parameterlist = [["kappa", -0.1, None],
                 ["gamma", 0.1, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = CoffeeCupPointModelDependent(parameters)

model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)

uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                features=None,
                                                save_figures=True,
                                                rosenblatt=False)

uncertainty.allParameters()
