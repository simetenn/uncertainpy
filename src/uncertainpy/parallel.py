import os
import subprocess
import traceback

import numpy as np
import multiprocess as mp
import scipy.interpolate as scpi

from utils import create_logger
from features import GeneralFeatures


"""
result = {"directComparison": {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                               "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature1d": {"U": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature0d": {"U": 1,
                        "t": None},
          "feature2d": {"U": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature_adaptive": {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                               "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                               "interpolation": <scipy.interpolate.fitpack2.InterpolatedUnivariateSpline object at 0x7f1c78f0d4d0>},
          "featureInvalid": {"U": None,
                             "t": None}}
"""


class Parallel:
    def __init__(self,
                 model,
                 features=None,
                 verbose_level="info",
                 verbose_filename=None):

        self._features = None
        self._model = None

        self.features = features
        self.model = model

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)



    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, new_features):
        self._features = new_features

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model


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

        if len(ut) != 0:
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
        result = {"feature1d": {"U": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                  "feature2d": {"U": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])},
                  "directComparison": {"U": array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
                                       "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                  "feature0d": {"U": 1}}
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

        for feature in features_0d:
            if feature in self.features.adaptive_features or \
                    (feature == "directComparison" and self.model.adaptive_model):
                raise AttributeError("{} is 0D,".format(feature)
                                     + " unable to perform interpolation")

        for feature in features_1d:
            if feature in self.features.adaptive_features or \
                    (feature == "directComparison" and self.model.adaptive_model):
                if results[feature]["t"] is None:
                    raise AttributeError("{} does not return any t values.".format(feature)
                                         + " Unable to perform interpolation")

                interpolation = scpi.InterpolatedUnivariateSpline(results[feature]["t"],
                                                                  results[feature]["U"],
                                                                  k=3)
                results[feature]["interpolation"] = interpolation


        for feature in features_2d:
            if feature in self.features.adaptive_features or \
                    (feature == "directComparison" and self.model.adaptive_model):
                raise NotImplementedError("{feature},".format(feature=feature)
                                          + " no support for >= 2D interpolation")

        return results



    def run(self, model_parameters):

        # Try-except to catch exeptions and print stack trace
        try:
            model_result = self.model.run(model_parameters)

            try:
                t, U = model_result
            except ValueError as error:
                msg = "model.run() must return t and U (return t, U | return None, U)"
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            if U is None:
                raise ValueError("U has not been calculated")

            if np.all(np.isnan(t)):
                t = None


            results = {}
            results["directComparison"] = {"t": t, "U": U}

            # Calculate features from the model results
            self.features.t = t
            self.features.U = U
            self.features.setup()
            feature_results = self.features.calculateFeatures()


            for feature in feature_results:
                feature_t = feature_results[feature]["t"]
                if feature_t is not None and np.all(np.isnan(feature_t)):
                    feature_t = None

                results[feature] = {"U": feature_results[feature]["U"],
                                    "t": feature_t}

            # Create interpolations
            results = self.createInterpolations(results)

            return results


        except Exception as e:
            print("Caught exception in parallel run of model:")
            print("")
            traceback.print_exc()
            print("")
            raise e
