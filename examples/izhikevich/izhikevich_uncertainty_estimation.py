import uncertainpy
from IzhikevichModel import IzhikevichModel

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = IzhikevichModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


features = uncertainpy.NeuronFeatures(features_to_run="all")

uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                features=features,
                                                save_figures=True)

uncertainty.UQ()
