import subprocess
import datetime

from memory import Memory
from uncertainty import UncertaintyEstimations
from IzhikevichModel import IzhikevichModel


original_parameters = {
    "a": 0.02,  # mS/cm2
    "b": 0.2,    # mS/cm2
    "c": -65,   # mS/cm2
    "d": 8
}

memory = Memory(10)
memory.start()

fitted_parameters = ["a", "b", "c", "d"]

model = IzhikevichModel()

percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
test_distributions = {"uniform": percentages}
exploration = UncertaintyEstimations(model, original_parameters, original_parameters, test_distributions,
                                     output_dir_data="data/izhikevich")
exploration.exploreParameters()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
