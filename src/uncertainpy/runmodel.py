from tqdm import tqdm
from xvfbwrapper import Xvfb

import numpy as np
import multiprocess as mp

import os
import subprocess
import scipy.interpolate as scpi
import traceback


from uncertainpy import Data
from uncertainpy.features import GeneralFeatures
from uncertainpy.utils import create_logger


"""
results = {'directComparison': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
           'feature2d': array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
           'feature1d': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
           'feature0d': 1}

solves = [results1, results2, ..., resultsN]
"""


class RunModel:
    def __init__(self,
                 model,
                 features=None,
                 CPUs=mp.cpu_count(),
                 supress_model_output=True,
                 supress_model_graphics=True,
                 verbose_level="info",
                 verbose_filename=None):


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

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


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
        features_0d, features_1d, features_2d = self.sortFeatures(solves[0])

        self.data.features_0d = features_0d
        self.data.features_1d = features_1d
        self.data.features_2d = features_2d

        if self.is_adaptive(solves) and not self.model.adaptive_model:
            # TODO if the model is adaptive perform the complete interpolation here instead.
            raise ValueError("The number of simulation points varies between simulations."
                             + " Try setting adaptive_model=True in model()")



        for feature in self.data.features_2d:
            if "interpolation" in solves[0][feature]:
                raise NotImplementedError("Feature: {feature},".format(feature=feature)
                                          + " no support for >= 2D interpolation")

            else:
                if "t" in solves[0][feature]:
                    self.data.t[feature] = solves[0][feature]["t"]
                else:
                    self.data.t[feature] = solves[0]["directComparison"]["t"]

                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature]["U"])


        for feature in self.data.features_1d:
            if "interpolation" in solves[0][feature]:
                ts = []
                interpolations = []
                for solved in solves:
                    if "t" in solved[feature]:
                        ts.append(solved[feature]["t"])
                    else:
                        ts.append(solved["directComparison"]["t"])

                    interpolations.append(solved[feature]["interpolation"])

                self.data.t[feature], self.data.U[feature] = self.performInterpolation(ts, interpolations)
            else:
                if "t" in solves[0][feature]:
                    self.data.t[feature] = solves[0][feature]["t"]
                else:
                    self.data.t[feature] = solves[0]["directComparison"]["t"]

                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature]["U"])

                # self.data.U[feature] = np.array(self.U[feature])


        for feature in self.data.features_0d:
            self.data.U[feature] = []
            self.data.t[feature] = None
            for solved in solves:
                self.data.U[feature].append(solved[feature]["U"])

            # self.U[feature] = np.array(self.U[feature])

        # self.t[feature] = np.array(self.t[feature])
        self.data.U[feature] = np.array(self.data.U[feature])

        print self.data
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
    def is_adaptive(self, solves):
        """
Test if solves is an adaptive result
        """
        for feature in self.data.features_1d + self.data.features_2d:
            u_prev = solves[0][feature]["U"]
            for solve in solves[1:]:
                u = solve[feature]["U"]
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


    def run_subprocess(self, parameters):
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
            self.logger.info(ut)

        if simulation.returncode != 0:
            self.logger.error(ut)
            raise RuntimeError(err)


        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        return t, U



    def sortFeatures(self, results):

        """
        result = {'feature1d': {'U': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                  'feature2d': {'U': array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])},
                  'directComparison': {'U': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
                                       't': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                  'feature0d': {'U': 1}}
        """

        features_2d = []
        features_1d = []
        features_0d = []

        for feature in results:
            if hasattr(results[feature]["U"], "__iter__"):
                if len(results[feature]["U"].shape) == 0:
                    features_0d.append(feature)
                elif len(results[feature]["U"].shape) == 1:
                    features_1d.append(feature)
                else:
                    features_2d.append(feature)
            else:
                features_0d.append(feature)

        return features_0d, features_1d, features_2d



    def createInterpolations(self, results):
        features_0d, features_1d, features_2d = self.sortFeatures(results)

        t = results["directComparison"]["t"]
        if t is None:
            raise AttributeError("Model does not return any t values."
                                 + " Unable to perform interpolation")


        for feature in features_0d:
            self.logger.warning("Feature: {feature} is 0D,".format(feature=feature)
                                + " unable to perform interpolation")
            continue

        for feature in features_1d:
            interpolation = scpi.InterpolatedUnivariateSpline(t,
                                                              results["directComparison"]["U"],
                                                              k=3)
            results[feature]["interpolation"] = interpolation


        for feature in features_2d:
            self.logger.warning("Feature: {feature},".format(feature=feature)
                                + " no support for >= 2D interpolation")
            continue

        return results



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

            # new_process = True
            if self.model.new_process:
                t, U = self.run_subprocess(parameters)

            else:
                model_result = self.model.run(parameters)

                if (not isinstance(model_result, tuple) or not isinstance(model_result, list)) and len(model_result) != 2:
                    raise RuntimeError("model.run() must return t and U (return t, U | return None, U)")

                t, U = model_result

                if U is None:
                    raise ValueError("U has not been calculated")

                if np.all(np.isnan(t)):
                    t = None



            # Calculate features from the model results

            results = {}

            self.features.t = t
            self.features.U = U
            feature_results = self.features.calculateFeatures()

            for feature in feature_results:
                results[feature] = {"U": feature_results[feature]}

            results["directComparison"] = {"t": t, "U": U}


            # Create interpolation
            if self.model.adaptive_model:
                results = self.createInterpolations(results)

            return results

        except Exception as e:
            print("Caught exception in evaluateNodeFunction:")
            print("")
            traceback.print_exc()
            print("")
            raise e
