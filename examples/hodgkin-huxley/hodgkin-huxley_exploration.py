import subprocess
import datetime
import uncertainpy
import chaospy as cp


orignal_parameters = [["V_rest", 0, None],
                      ["Cm", 1, cp.Uniform(0.8, 1.5)],
                      ["gbar_Na", 120, cp.Uniform(65, 260)],
                      ["gbar_K", 36, cp.Uniform(26, 49)],
                      ["gbar_l", 0.3, cp.Uniform(0.13, 0.5)],
                      ["E_Na", 115, cp.Uniform(95, 119)],
                      ["E_K", -12, cp.Uniform(-9, -14)],
                      ["E_l", 10.613, cp.Uniform(4, 22)]]




parameters = uncertainpy.Parameters(orignal_parameters)

model = uncertainpy.HodkinHuxleyModel(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)

features = uncertainpy.NeuronFeatures(features_to_run="all")


exploration = uncertainpy.UncertaintyEstimations(model,
                                                 CPUs=7,
                                                 features=features,
                                                 save_figures=True)

percentages = [0.01, 0.03]
test_distributions = {"uniform": percentages}
exploration.exploreParameters(test_distributions)

mc_samples = [10, 100, 1000]
exploration.compareMC(mc_samples)


print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
