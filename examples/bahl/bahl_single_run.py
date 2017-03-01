import uncertainpy as un
import chaospy as cp

from NeuronModelBahl import NeuronModelBahl

parameterlist = [["e_pas", -80, cp.Uniform(-60, -85)],
                 ["apical Ra", 261, cp.Uniform(150, 300)]]

model = NeuronModelBahl(parameters=parameterlist)

uncertainty = un.UncertaintyEstimation(model, save_figures=True)

uncertainty.UQ(plot_condensed=False, plot_simulator_results=True)
uncertainty.UQ(single=True, plot_simulator_results=True)
