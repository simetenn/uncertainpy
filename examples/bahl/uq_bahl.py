import uncertainpy as un
import chaospy as cp

parameterlist = [["e_pas", -80, cp.Uniform(-60, -85)],
                 ["apical Ra", 261, cp.Uniform(150, 300)]]


model = un.NeuronModel(model_path="bahl_neuron_model",
                       adaptive_model=True)

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameterlist,
                                       save_figures=True)

uncertainty.uncertainty_quantification(plot_condensed=False,
                                       plot_simulator_results=True)
uncertainty.uncertainty_quantification(single=True,
                                       plot_simulator_results=True)
