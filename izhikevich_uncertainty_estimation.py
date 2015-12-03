import subprocess
import datetime

from memory import Memory
from uncertainty import UncertaintyEstimations
from IzhikevichModel import IzhikevichModel


memory = Memory(10)
memory.start()

model = IzhikevichModel()

percentages = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
test_distributions = {"uniform": percentages}

exploration = UncertaintyEstimations(model, test_distributions, output_dir_data="data/izhikevich")
exploration.exploreParameters()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
