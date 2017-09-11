import uncertainpy as un
import chaospy as cp

parameterlist = [["e_pas", -80, cp.Uniform(-60, -85)],
                 ["apical Ra", 261, cp.Uniform(150, 300)]]


model = un.NeuronModel(path="bahl_neuron_model", name="bahl")

features = un.SpikingFeatures()

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameterlist,
                                       features=features,
                                       save_figures=True)

uncertainty.uncertainty_quantification(plot_condensed=False,
                                       plot_model_results=True)
uncertainty.uncertainty_quantification(single=True,
                                       plot_model_results=False)
