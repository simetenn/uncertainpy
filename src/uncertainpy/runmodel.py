from tqdm import tqdm
from xvfbwrapper import Xvfb

import numpy as np
import multiprocess as mp


from data import Data
from features import GeneralFeatures
from utils import create_logger
from parallel import Parallel


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

solves = [results 1, results 2, ..., results N]
"""


class RunModel(object):
    def __init__(self,
                 model,
                 features=None,
                 CPUs=mp.cpu_count(),
                 supress_model_graphics=True,
                 verbose_level="info",
                 verbose_filename=None):

        self._model = None
        self._features = None

        self.data = Data()
        self.parallel = Parallel(model)

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        self.model = model

        self.CPUs = CPUs
        self.supress_model_graphics = supress_model_graphics



        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, new_features):
        self._features = new_features
        self.parallel.features = new_features


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        if callable(new_model):
            print "new_model is callable"

        self._model = new_model
        self.parallel.model = new_model

        if new_model is not None:
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
        features_0d, features_1d, features_2d = self.parallel.sortFeatures(solves[0])

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

                self.data.t[feature], self.data.U[feature] = \
                    self.performInterpolation(ts, interpolations)
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
            self.data.t[feature] = solves[0][feature]["t"]
            self.data.U[feature] = []
            for solved in solves:
                self.data.U[feature].append(solved[feature]["U"])

            # self.U[feature] = np.array(self.U[feature])


        self.data.removeOnlyInvalidResults()




    def create_model_parameters(self, nodes, uncertain_parameters):
        model_parameters = []
        for node in nodes.T:
            if isinstance(node, float) or isinstance(node, int):
                node = [node]

            # New setparameters
            parameters = {}
            for j, parameter in enumerate(uncertain_parameters):
                parameters[parameter] = node[j]

            model_parameters.append(parameters)

        return model_parameters


        # model_parameters = []
        # for node in nodes.T:
        #     if isinstance(node, float) or isinstance(node, int):
        #         node = [node]
        #
        #     # New setparameters
        #     parameters = {}
        #     for j, parameter in enumerate(uncertain_parameters):
        #         parameters[parameter] = node[j]
        #
        #     model_parameters.append(parameters)
        #
        # return model_parameters


    def evaluateNodes(self, nodes):

        if self.supress_model_graphics:
            vdisplay = Xvfb()
            vdisplay.start()

        solves = []
        pool = mp.Pool(processes=self.CPUs)

        model_parameters = self.create_model_parameters(nodes, self.data.uncertain_parameters)
        for result in tqdm(pool.imap(self.parallel.run, model_parameters),
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
