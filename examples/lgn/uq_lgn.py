import uncertainpy as un

path = "dLGN_modelDB/"


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

parameterlist = [["cap", 1.1, None],
                 ["Rm", 22000, None],
                 ["Vrest", -63, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.uniform(0.05))

model = un.NeuronModel(path=path, adaptive=True)

# features = un.SpikingFeatures(features_to_run="all")

features = un.EfelFeatures(features_to_run="Spikecount")

print features.features_to_run
import sys
sys.exit(1)
uncertainty = un.UncertaintyEstimation(model,
                                       parameters=parameters,
                                       features=features,
                                       CPUs=7,
                                       allow_incomplete=True)

uncertainty.uncertainty_quantification()
