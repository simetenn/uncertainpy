import uncertainpy
from IzhikevichModel import IzhikevichModel


parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = IzhikevichModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(1).uniform)

features = uncertainpy.SpikingFeatures(features_to_run="all")

exploration = uncertainpy.UncertaintyEstimations(model,
                                                 CPUs=7,
                                                 seed=10,
                                                 features=features,
                                                 save_figures=True,
                                                 single_parameter_runs=True,
                                                 rosenblatt=False)

percentages = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

mc_samples = [10, 100, 1000]
exploration.compareMC(mc_samples)
