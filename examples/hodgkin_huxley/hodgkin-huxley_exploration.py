import uncertainpy
from HodgkinHuxley import HodgkinHuxley

orignal_parameters = [["V_rest", -65, None],
                      ["Cm", 1, None],
                      ["gbar_Na", 120, None],
                      ["gbar_K", 36, None],
                      ["gbar_l", 0.3, None],
                      ["E_Na", 50, None],
                      ["E_K", -77, None],
                      ["E_l", -50.613, None]]




parameters = uncertainpy.Parameters(orignal_parameters)

model = HodgkinHuxley(parameters)
model.set_all_distributions(uncertainpy.Distribution(0.2).uniform)

features = uncertainpy.SpikingFeatures(features_to_run="all")

exploration = uncertainpy.UncertaintyQuantifications(model,
                                                 supress_model_output=False,
                                                 features=features,
                                                 plot=True)

percentages = [0.01, 0.03]
test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

mc_samples = [10, 100, 1000]
exploration.compareMC(mc_samples)
