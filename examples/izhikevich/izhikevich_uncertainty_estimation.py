import uncertainpy

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.IzhikevichModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


features = uncertainpy.NeuronFeatures(features_to_run="all")

uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                features=features,
                                                save_figures=True)

uncertainty.allParameters()
