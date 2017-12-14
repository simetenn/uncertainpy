import sys
import os
import traceback
import subprocess
import scipy

import numpy as np
import multiprocess as mp

# TODO see if neuron models can be run directly from this class


__all__ = ["evaluateNodeFunction"]
__version__ = "0.1"

filepath = os.path.abspath(__file__)
filedir = os.path.dirname(filepath)

# In it's own function since the multiprocessing library is unable to picle
# methods in a class.
def evaluateNodeFunction(data):
    """
data = {"model_cmds": model_cmds
        "supress_model_output": supress_model_output
        "adaptive_model": adaptive_model
        "node": node
        "uncertain_parameters: uncertain_parameters
        "features_cmds": features_cmds
        "features_kwargs": features_kwargs}
    """

    # Try-except to catch exeptions and print stack trace
    try:
        model_cmds = data["model_cmds"]
        supress_model_output = data["supress_model_output"]
        adaptive_model = data["adaptive_model"]
        new_process = data["new_process"]
        node = data["node"]
        uncertain_parameters = data["uncertain_parameters"]
        features_cmds = data["features_cmds"]
        features_kwargs = data["features_kwargs"]

        if isinstance(node, float) or isinstance(node, int):
            node = [node]

        # New setparameters
        parameters = {}
        j = 0
        for parameter in uncertain_parameters:
            parameters[parameter] = node[j]
            j += 1

        new_process = True
        if new_process:


            current_process = mp.current_process().name.split("-")
            if current_process[0] == "PoolWorker":
                current_process = str(current_process[-1])
            else:
                current_process = "0"

            model_cmds += ["--CPU", current_process,
                           "--save_path", filedir,
                           "--parameters"]

            for parameter in parameters:
                model_cmds.append(parameter)
                model_cmds.append("{:.16f}".format(parameters[parameter]))



            simulation = subprocess.Popen(model_cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
            ut, err = simulation.communicate()

            if not supress_model_output and len(ut) != 0:
                print(ut)


            if simulation.returncode != 0:
                print(ut)
                raise RuntimeError(err)


            values = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
            t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

            os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
            os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        else:
            pass

            # filedir =
            # sys.path.insert(0, file_dir)
            # module = __import__(file_name.split(".")[0])
            # model = getattr(module, args.model_name)
            #
            # model_kwargs = dict(zip(args.model_kwargs[::2], args.model_kwargs[1::2]))
            # simulation = model(**model_kwargs)

        # Calculate features from the model results
        results = {}

        # TODO Should t be stored for all results? Or should none be used for features

        sys.path.insert(0, features_cmds["filedir"])
        module = __import__(features_cmds["filename"].split(".")[0])
        features = getattr(module, features_cmds["feature_name"])(t=time, values=U, **features_kwargs)

        feature_results = features.calculate_features()

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

                interpolation = scipy.interpolate.InterpolatedUnivariateSpline(time, values, k=3)
                results["directComparison"] = (time, values, interpolation)

            else:
                raise NotImplementedError("No support yet for >= 2D interpolation")


        else:
            if np.all(np.isnan(t)):
                results["directComparison"] = (None, U, None)
            else:
                results["directComparison"] = (time, values, None)

        # All results are saved as {feature: (x, U, interpolation)} as appropriate.

        return results

    except Exception as e:
        print("Caught exception in evaluateNodeFunction:")
        print("")
        traceback.print_exc()
        print("")
        raise e
