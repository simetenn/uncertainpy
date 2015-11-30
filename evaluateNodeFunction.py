import sys
import subprocess
import scipy
import os

import numpy as np
import multiprocessing as mp


def evaluateNodeFunction(data):
    """
    all_data = (cmds, node, tmp_parameter_names, modelfile, modelpath, features)
    """

    cmd = data[0]
    node = data[1]
    tmp_parameter_names = data[2]
    features = data[3]

    if isinstance(node, float) or isinstance(node, int):
            node = [node]

    # New setparameters
    tmp_parameters = {}
    j = 0
    for parameter in tmp_parameter_names:
        tmp_parameters[parameter] = node[j]
        j += 1

    current_process = mp.current_process().name.split("-")
    if current_process[0] == "PoolWorker":
        current_process = str(current_process[-1])
    else:
        current_process = "0"

    cmd = cmd + ["--CPU", current_process]

    for parameter in tmp_parameters:
        cmd.append(parameter)
        cmd.append(str(tmp_parameters[parameter]))

    simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ut, err = simulation.communicate()

    if simulation.returncode != 0:
        print "Error when running simulation:"
        print err
        sys.exit(1)


    V = np.load("tmp_U_%s.npy" % current_process)
    t = np.load("tmp_t_%s.npy" % current_process)

    os.remove("tmp_U_%s.npy" % current_process)
    os.remove("tmp_t_%s.npy" % current_process)


    # TODO Do a feature selection here. Make it so several feature
    # selections are performed at this step.
    for feature in features:
        pass

    interpolation = scipy.interpolate.InterpolatedUnivariateSpline(t, V, k=3)

    return (t, V, interpolation)
