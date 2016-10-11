import sys
import subprocess
import scipy
import os

import numpy as np
import multiprocessing as mp

# TODO see if neuron models can be run directly from this class


__all__ = ["evaluateNodeFunction"]
__version__ = "0.1"

filepath = os.path.abspath(__file__)
filedir = os.path.dirname(filepath)


def evaluateNodeFunction(data):
    """
all_data = (model_cmds,
            supress_model_output,
            adaptive_model,
            node,
            tmp_parameter_names,
            feature_cmds,
            feature_kwargs)
    """
    
    model_cmds = data[0]
    supress_model_output = data[1]
    adaptive_model = data[2]
    node = data[3]
    tmp_parameter_names = data[4]
    feature_cmds = data[5]
    feature_kwargs = data[6]

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

    model_cmds += ["--CPU", current_process,
                   "--save_path", filedir,
                   "--parameters"]

    for parameter in tmp_parameters:
        model_cmds.append(parameter)
        model_cmds.append("{:.16f}".format(tmp_parameters[parameter]))


    simulation = subprocess.Popen(model_cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ut, err = simulation.communicate()

    if not supress_model_output and len(ut) != 0:
        print ut


    if simulation.returncode != 0:
        print ut
        raise RuntimeError(err)


    U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
    t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

    os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
    os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))



    # Calculate features from the model results
    results = {}

    # TODO Should t be stored for all results? Or should none be used for features

    sys.path.insert(0, feature_cmds[0])
    module = __import__(feature_cmds[1].split(".")[0])
    features = getattr(module, feature_cmds[2])(t=t, U=U, **feature_kwargs)

    feature_results = features.calculateFeatures()

    for feature in feature_results:
        tmp_result = feature_results[feature]

        if tmp_result is None:
            results[feature] = (None, np.nan, None)

        # elif adaptive_model:
        #     if len(U.shape) == 0:
        #         raise AttributeError("Model returns a single value, unable to perform interpolation")
        #
        #     if len(tmp_result.shape) == 0:
        #         # print "Warning: {} returns a single number, no interpolation performed".format(feature)
        #         results[feature] = (None, tmp_result, None)
        #
        #     elif len(tmp_result.shape) == 1:
        #         if np.all(np.isnan(t)):
        #             raise AttributeError("Model does not return any t values. Unable to perform interpolation")
        #
        #         interpolation = scipy.interpolate.InterpolatedUnivariateSpline(t, tmp_result, k=3)
        #         results[feature] = (t, tmp_result, interpolation)
        #
        #     else:
        #         raise NotImplementedError("Error: No support yet for >= 2d interpolation")

        else:
            if np.all(np.isnan(t)):
                results[feature] = (None, tmp_result, None)
            else:
                results[feature] = (t, tmp_result, None)
        # results[feature] = (t, tmp_result, None)


    # Create interpolation
    if adaptive_model:
        if len(U.shape) == 0:
            raise RuntimeWarning("Model returns a single value, unable to perform interpolation")

        elif len(U.shape) == 1:
            if np.all(np.isnan(t)):
                raise AttributeError("Model does not return any t values. Unable to perform interpolation")

            interpolation = scipy.interpolate.InterpolatedUnivariateSpline(t, U, k=3)
            results["directComparison"] = (t, U, interpolation)

        else:
            raise NotImplementedError("No support yet for >= 2d interpolation")


    else:
        if np.all(np.isnan(t)):
            results["directComparison"] = (None, U, None)
        else:
            results["directComparison"] = (t, U, None)

    # All results are saved as {feature : (x, U, interpolation)} as appropriate.

    return results
