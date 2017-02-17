import uncertainpy as un
import chaospy as cp

from NeuronModelBahl import NeuronModelBahl

parameterlist = [["apical Ra", 261, cp.Uniform(150, 300)],
                 ["soma Ra", 82, cp.Uniform(80, 200)]]


parameters = un.Parameters(parameterlist)
model = NeuronModelBahl(parameters=parameters)

uncertainty = un.UncertaintyEstimation(model,
                                       save_figures=True,
                                       plot_type="all",
                                       plot_simulator_results=False)

uncertainty.UQ()

# Set plot_simulator_results to false so simulator results is not ploted
uncertainty.UQ(single=True)
