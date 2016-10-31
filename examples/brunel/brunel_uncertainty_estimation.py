import uncertainpy
from BrunelNetworkModel import BrunelNetworkModel


parameterlist = [["J_E", 4, None],
                 ["g", 4, None]]

parameters = uncertainpy.Parameters(parameterlist)

model = BrunelNetworkModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


exploration = uncertainpy.UncertaintyEstimation(model,
                                                CPUs=8,
                                                features=None,
                                                save_figures=True,
                                                figureformat=".pdf")

exploration.allParameters()
