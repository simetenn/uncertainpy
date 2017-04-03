import uncertainpy as un

model_path = "dLGN_modelDB/"


parameterlist = [["cap", 1.1, None],
                 ["Rm", 22000, None],
                 ["Vrest", -63, None],
                 ["Epas", -67, None],
                 ["gna", 0.09, None],
                 ["nash", -52.6, None],
                 ["gkdr", 0.37, None],
                 ["kdrsh", -51.2, None],
                 ["gahp", 6.4e-5, None],
                 ["gcat", 1.17e-5, None]]


parameters = un.Parameters(parameterlist)
parameters.setAllDistributions(un.Distribution(0.05).uniform)

model = un.NeuronModel(model_path=model_path,
                       adaptive_model=True)

features = un.NeuronFeatures(features_to_run="all")

uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features)

uncertainty.UQ()
