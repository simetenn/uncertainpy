import subprocess
import datetime

from memory import Memory
from uncertainty import UncertaintyEstimations
from CoffeeCupPointModel import CoffeeCupPointModel


data_dir = "data/"
output_figures_dir = "figures/"
figureformat = ".png"
output_gif_dir = "gifs/"

original_parameters = {
    "kappa": -0.01,
    "u_env": 20
}

memory = Memory(10)
memory.start()

#fitted_parameters = ["kappa", "u_env"]
fitted_parameters = ["kappa"]

model = CoffeeCupPointModel()


test_distributions = {"uniform": [0.02, 0.03]}
exploration = UncertaintyEstimations(model, original_parameters, fitted_parameters, test_distributions,
                                     output_dir_data="data/coffee", CPUs=1)
exploration.exploreParameters()

memory.end()

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
