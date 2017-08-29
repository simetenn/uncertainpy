import uncertainpy as un
from izhikevich_function import izhikevich

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]

parameters = un.Parameters(parameterlist)
parameters.set_all_distributions(un.uniform(0.5))


model = un.Model(izhikevich, labels=["time [ms]", "voltage [mv]"])

features = un.SpikingFeatures(features_to_run="all")

uncertainty = un.UncertaintyEstimation(model=model,
                                       parameters=parameters,
                                       features=features)

uncertainty.uncertainty_quantification()
