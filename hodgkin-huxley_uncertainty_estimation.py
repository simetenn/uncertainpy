import subprocess
import datetime

from memory import Memory
from uncertainty import UncertaintyEstimations, UncertaintyEstimation
from distribution import Distribution
from parameters import Parameters
from HodkinHuxleyModel import HodkinHuxleyModel


data_dir = "data/"
output_figures_dir = "figures/"
figureformat = ".png"
output_gif_dir = "gifs/"

original_parameters = {
    "gbar_Na": 120,  # mS/cm2
    "gbar_K": 36,    # mS/cm2
    "gbar_l": 0.3,   # mS/cm2
}

memory = Memory(10)
memory.start()

fitted_parameters = ["gbar_Na", "gbar_K", "gbar_l"]

model = HodkinHuxleyModel()


test_distributions = {"uniform": [0.02, 0.03]}
exploration = UncertaintyEstimations(model, original_parameters, fitted_parameters, test_distributions,
                                     output_dir_data="data/hodgkin-huxley", CPUs=1)
exploration.exploreParameters()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
