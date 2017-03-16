from tqdm import tqdm
from xvfbwrapper import Xvfb

import numpy as np
import multiprocess as mp
import os
import subprocess
import scipy
import traceback
import sys

from uncertainpy import Data
from uncertainpy.features import GeneralFeatures

class RunModel:
    def __init__(self,
                 model,
                 features=None,
                 CPUs=mp.cpu_count(),
                 supress_model_output=True,
                 supress_model_graphics=True):


        self.model = model

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        self.CPUs = CPUs

        self.supress_model_graphics = supress_model_graphics
        self.supress_model_output = supress_model_output

        self.data = Data()

        self.set_model(model)


    def set_model(self, model):

        self.model = model

        if model is not None:
            self.data.xlabel = self.model.xlabel
            self.data.ylabel = self.model.ylabel


    def performInterpolation(self, ts, interpolation):
        lengths = []
        for s in ts:
            lengths.append(len(s))

        index_max_len = np.argmax(lengths)
        t = ts[index_max_len]

        interpolated_solves = []
        for inter in interpolation:
            interpolated_solves.append(inter(t))

        interpolated_solves = np.array(interpolated_solves)

        return t, interpolated_solves




    def storeResults(self, solves):

        self.data.setFeatures(solves[0])


        if self.isSolvesAdaptive(solves) and not self.model.adaptive_model:
            # TODO if the model is adaptive perform the complete interpolation here instead.
            raise ValueError("The number of simulation points varies between simulations. Try setting adaptive_model=True in model()")



        for feature in self.data.features_2d:
            if self.model.adaptive_model and feature == "directComparison":
                raise NotImplementedError("Support for >= 2d interpolation is not yet implemented")

            else:
                self.data.t[feature] = solves[0][feature][0]

                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature][1])

                # self.U[feature] = np.array(self.U[feature])

        for feature in self.data.features_1d:
            if self.model.adaptive_model and feature == "directComparison":
                ts = []
                interpolation = []
                for solved in solves:
                    ts.append(solved[feature][0])
                    interpolation.append(solved[feature][2])

                self.data.t[feature], self.data.U[feature] = self.performInterpolation(ts, interpolation)
            else:
                self.data.t[feature] = solves[0][feature][0]
                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature][1])

                # self.data.U[feature] = np.array(self.U[feature])


        for feature in self.data.features_0d:
            self.data.U[feature] = []
            self.data.t[feature] = None
            for solved in solves:
                self.data.U[feature].append(solved[feature][1])

            # self.U[feature] = np.array(self.U[feature])

        # self.t[feature] = np.array(self.t[feature])
        self.data.U[feature] = np.array(self.data.U[feature])

        self.data.removeOnlyInvalidResults()


    def evaluateNodes(self, nodes):

        if self.supress_model_graphics:
            vdisplay = Xvfb()
            vdisplay.start()

        solves = []
        pool = mp.Pool(processes=self.CPUs)

        for result in tqdm(pool.imap(self.evaluateNodeFunction, nodes.T),
                           desc="Running model",
                           total=len(nodes.T)):


            solves.append(result)


        pool.close()

        if self.supress_model_graphics:
            vdisplay.stop()

        return np.array(solves)



    # TODO should this check one specific feature.
    # Return false for all features?
    def isSolvesAdaptive(self, solves):
        """
Test if solves is an adaptive result
        """
        for feature in self.data.features_1d + self.data.features_2d:
            u_prev = solves[0][feature][1]
            for solve in solves[1:]:
                u = solve[feature][1]
                if u_prev.shape != u.shape:
                    return True
                u_prev = u
        return False



    def run(self, nodes, uncertain_parameters):

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        self.data.uncertain_parameters = uncertain_parameters

        solves = self.evaluateNodes(nodes)
        self.storeResults(solves)

        return self.data


    def evaluateNodeFunction(self, node):

        # Try-except to catch exeptions and print stack trace
        try:

            if isinstance(node, float) or isinstance(node, int):
                node = [node]

            # New setparameters
            parameters = {}
            j = 0
            for parameter in self.data.uncertain_parameters:
                parameters[parameter] = node[j]
                j += 1

            new_process = True
            if new_process:
                current_process = mp.current_process().name.split("-")
                if current_process[0] == "PoolWorker":
                    current_process = str(current_process[-1])
                else:
                    current_process = "0"

                filepath = os.path.abspath(__file__)
                filedir = os.path.dirname(filepath)

                model_cmds = self.model.cmd()
                model_cmds += ["--CPU", current_process,
                               "--save_path", filedir,
                               "--parameters"]

                for parameter in parameters:
                    model_cmds.append(parameter)
                    model_cmds.append("{:.16f}".format(parameters[parameter]))



                simulation = subprocess.Popen(model_cmds,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE,
                                              env=os.environ.copy())
                ut, err = simulation.communicate()

                if not self.supress_model_output and len(ut) != 0:
                    print ut


                if simulation.returncode != 0:
                    print ut
                    raise RuntimeError(err)


                U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
                t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

                os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
                os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

            else:
                t, U = self.model.run()
                pass



            # Calculate features from the model results
            results = {}

            # TODO Should t be stored for all results? Or should none be used for features

            self.features.t = t
            self.features.U = U
            feature_results = self.features.calculateFeatures()

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
            if self.model.adaptive_model:
                if len(U.shape) == 0:
                    raise RuntimeWarning("Model returns a single value, unable to perform interpolation")

                elif len(U.shape) == 1:
                    if np.all(np.isnan(t)):
                        raise AttributeError("Model does not return any t values. Unable to perform interpolation")

                    interpolation = scipy.interpolate.InterpolatedUnivariateSpline(t, U, k=3)
                    results["directComparison"] = (t, U, interpolation)

                else:
                    raise NotImplementedError("No support yet for >= 2D interpolation")


            else:
                if np.all(np.isnan(t)):
                    results["directComparison"] = (None, U, None)
                else:
                    results["directComparison"] = (t, U, None)

            # All results are saved as {feature: (x, U, interpolation)} as appropriate.

            return results

        except Exception as e:
            print("Caught exception in evaluateNodeFunction:")
            print("")
            traceback.print_exc()
            print("")
            raise e
