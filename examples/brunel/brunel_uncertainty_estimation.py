import uncertainpy as un
from BrunelNetworkModel import BrunelNetworkModel


parameterlist = [["J_E", 4, None],
                 ["g", 4, None]]

model = BrunelNetworkModel(parameterlist)
model.setAllDistributions(un.Distribution(0.5).uniform)


uncertainty = un.UncertaintyEstimation(model, features=None)

uncertainty.UQ(plot_condensed=False, plot_simulator_results=True)
