import datetime
import uncertainpy
import chaospy as cp

from NeuronModelBahl import NeuronModelBahl

parameterlist = [["apical Ra", 261, cp.Uniform(150, 300)],
                 ["soma Ra", 82, cp.Uniform(80, 200)]]#,
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()],
                #  ["", , cp.Uniform()]]

parameters = uncertainpy.Parameters(parameterlist)
model = NeuronModelBahl(parameters=parameters)

uncertainty = uncertainpy.UncertaintyEstimation(model, CPUs=1, save_figures=True)


uncertainty.singleParameters()
uncertainty.allParameters()
uncertainty.plot.plotSimulatorResults()

print "The total runtime is: " + str(datetime.timedelta(seconds=(uncertainty.timePassed())))
