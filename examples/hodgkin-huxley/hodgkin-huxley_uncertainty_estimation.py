import uncertainpy
import chaospy as cp

from HodgkinHuxleyModel import HodgkinHuxleyModel

def cdf(x, a, b, c, d):
    if x <= b:
        return (x-a)/(b-a + d-c)
    if b < x < c:
        return (b-a)/float((b-a + d-c))
    if x >= c:
        return (x-c+b)/(b-a + d-c)


def bnd(x, a, b, c, d):
    return(a, d)


def ppf(self, q, a, b, c, d):
    return (q < .5)*(2*q*(b-a) + a) + (q >= .5)*((2*q-1)*(d-c) + c)


Distribution = cp.construct(cdf=cdf, bnd=bnd, ppf=ppf)
distribution = Distribution(a=65, b=90, c=120, d=260)


parameters = [["V_rest", -65, None],
              ["Cm", 1, cp.Uniform(0.8, 1.5)],
              ["gbar_Na", 120, cp.Uniform(80, 160)],
              ["gbar_K", 36, cp.Uniform(26, 49)],
              ["gbar_l", 0.3, cp.Uniform(0.13, 0.5)],
              ["E_Na", 50, cp.Uniform(30, 54)],
              ["E_K", -77, cp.Uniform(-74, -79)],
              ["E_l", -50.613, cp.Uniform(-61, -43)]]

parameters = uncertainpy.Parameters(parameters)

model = HodgkinHuxleyModel(parameters=parameters)
model.setAllDistributions(uncertainpy.Distribution(0.2).uniform)

features = uncertainpy.NeuronFeatures(features_to_run="all")

exploration = uncertainpy.UncertaintyEstimation(model,
                                                seed=10,
                                                features=features,
                                                CPUs=7,
                                                save_figures=True,
                                                rosenblatt=True,
                                                figureformat=".pdf")

exploration.allParameters()
