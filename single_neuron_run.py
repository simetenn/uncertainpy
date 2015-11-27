import subprocess
import os
import sys
import numpy as np
import pylab as plt


from prettyPlot import prettyPlot

def singleNeuronRun(parameters, modelfile, modelpath):
    cmd = ["python", "simulation.py", modelfile, modelpath,
           "--CPU", "0"]

    for parameter in parameters:
        cmd.append(parameter)
        cmd.append(str(parameters[parameter]))

    simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ut, err = simulation.communicate()

    if simulation.returncode != 0:
        print "Error when running simulation:"
        print err
        sys.exit(1)


    V = np.load("tmp_U_0.npy")
    t = np.load("tmp_t_0.npy")

    os.remove("tmp_U_0.npy")
    os.remove("tmp_t_0.npy")

    return t, V

if __name__ == "__main__":

    modelfile = "INmodel.hoc"
    modelpath = "neuron_models/dLGN_modelDB/"

    parameters = {"Rm": 22200}
    _, V_max = singleNeuronRun(parameters, modelfile, modelpath)


    parameters = {"Rm": 22000}
    t, V = singleNeuronRun(parameters, modelfile, modelpath)
    ax, _ = prettyPlot(t, V, "Voltage, membrane resistance", "time", "voltage", color=0)
    ax.set_ylim([min(V_max), max(V_max)])

    plt.savefig("model_response_original.png")

    parameters = {"Rm": 21800}
    t, V = singleNeuronRun(parameters, modelfile, modelpath)
    prettyPlot(t, V, "Voltage, membrane resistance", "time", "voltage", color=2, new_figure=False)

    parameters = {"Rm": 22200}
    t, V = singleNeuronRun(parameters, modelfile, modelpath)
    prettyPlot(t, V, "Voltage, membrane resistance", "time", "voltage", color=4, new_figure=False)


    plt.savefig("model_response.png")
