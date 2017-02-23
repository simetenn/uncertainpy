import uncertainpy as un
import chaospy as cp

from NeuronModelBahl import NeuronModelBahl

parameterlist = [["apical Ra", 261, cp.Uniform(150, 300)],
                 ["decay_kfast", 55.581656, cp.Uniform(4, 660)]]


parameters = un.Parameters(parameterlist)
model = NeuronModelBahl(parameters=parameters)

uncertainty = un.UncertaintyEstimation(model,
                                       save_figures=True)

uncertainty.UQ(plot_condensed=False, plot_simulator_results=True)

# # Set plot_simulator_results to false so simulator results is not ploted
uncertainty.UQ(single=True, plot_simulator_results=True)
