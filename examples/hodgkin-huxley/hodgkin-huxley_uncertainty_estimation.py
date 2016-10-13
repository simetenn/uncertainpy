import uncertainpy
import chaospy as cp


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


orignal_parameters = [["V_rest", 0, None],
                      ["Cm", 1, cp.Uniform(0.8, 1.5)],
                      ["gbar_Na", 120, cp.Uniform(64, 260)],
                      ["gbar_K", 36, cp.Uniform(26, 49)],
                      ["gbar_l", 0.3, cp.Uniform(0.13, 0.5)],
                      ["E_Na", 115, cp.Uniform(95, 119)],
                      ["E_K", -12, cp.Uniform(-9, -14)],
                      ["E_l", 10.613, cp.Uniform(4, 22)]]

parameters = uncertainpy.Parameters(orignal_parameters)

model = uncertainpy.HodkinHuxleyModel(parameters)
features = uncertainpy.NeuronFeatures(features_to_run="all")

exploration = uncertainpy.UncertaintyEstimation(model,
                                                features=features,
                                                CPUs=7,
                                                save_figures=True)

exploration.allParameters()
