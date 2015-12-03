import subprocess
import datetime

import numpy as np

from memory import Memory
from uncertainty import UncertaintyEstimations
from IzhikevichModel import IzhikevichModel
from parameters import Parameters


memory = Memory(10)
memory.start()

parameterlist = [["a", 0.02, None],
                 ["b", 0.2, None],
                 ["c", -65, None],
                 ["d", 8, None]]


parameters = Parameters(parameterlist)
model = IzhikevichModel(parameters)



#percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
percentages = np.linspace(0.01, 0.25, 50)
test_distributions = {"uniform": percentages}

exploration = UncertaintyEstimations(model, test_distributions, output_dir_data="data/izhikevich")
exploration.exploreParameters()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
