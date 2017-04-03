import uncertainpy as un
import chaospy as cp

parameterlist = [["e_pas", -80, cp.Uniform(-60, -85)],
                 ["apical Ra", 261, cp.Uniform(150, 300)]]


model = un.NeuronModel(model_path="bahl_neuron_model",
                       adaptive_model=True)

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameterlist,
                                       save_figures=True)

uncertainty.UQ(plot_condensed=False,
               plot_simulator_results=True)
uncertainty.UQ(single=True,
               plot_simulator_results=True)
