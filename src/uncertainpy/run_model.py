from tqdm import tqdm
from xvfbwrapper import Xvfb

import numpy as np
import multiprocess as mp


from data import Data
from features import GeneralFeatures
from models import Model
from utils import create_logger
from parallel import Parallel
from parameters import Parameters

from base import ParameterBase


"""
result = {self.model.name: {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                               "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature1d": {"U": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature0d": {"U": 1,
                        "t": np.nan},
          "feature2d": {"U": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature_adaptive": {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                               "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                               "interpolation": <scipy.interpolate.fitpack2.InterpolatedUnivariateSpline object at 0x7f1c78f0d4d0>},
          "feature_invalid": {"U": np.nan,
                              "t": np.nan}}

results = [result 1, result 2, ..., result N]
"""


class RunModel(ParameterBase):
    def __init__(self,
                 model,
                 parameters,
                 features=None,
                 verbose_level="info",
                 verbose_filename=None,
                 CPUs=mp.cpu_count(),
                 supress_model_graphics=True):


        self.data = Data()
        self.parallel = Parallel(model=model,
                                 features=features,
                                 verbose_level=verbose_level,
                                 verbose_filename=verbose_filename)

        super(RunModel, self).__init__(model=model,
                                       parameters=parameters,
                                       features=features,
                                       verbose_level=verbose_level,
                                       verbose_filename=verbose_filename)

        self.CPUs = CPUs
        self.supress_model_graphics = supress_model_graphics



    @ParameterBase.features.setter
    def features(self, new_features):
        # Remove all labels but the model labels from data
        if self.model is not None and self.model.name in self.data.labels:
            if len(self.model.labels) > 0:
                self.data.labels = {self.model.name: self.model.labels}
            else:
                self.data.labels = {}

        ParameterBase.features.fset(self, new_features)

        self.parallel.features = self.features

        if self.features is not None:
            # Update data feature labels only if there are labels belonging to that feature
            for feature in self.features.labels:
                if len(self.features.labels[feature]) > 0:
                    self.data.labels[feature] = self.features.labels[feature]

            self.data.labels.update(self.features.labels)


    @ParameterBase.model.setter
    def model(self, new_model):
        # If model labels are in data remove them
        if self.model is not None and self.model.name in self.data.labels:
            del self.data.labels[self.model.name]

        ParameterBase.model.fset(self, new_model)

        self.parallel.model = self.model

        if self.model is not None:
            if self.model.labels:
                self.data.labels[self.model.name] = self.model.labels

            self.data.model_name = self.model.name


    def perform_interpolation(self, ts, interpolation):
        lengths = []
        for s in ts:
            lengths.append(len(s))

        index_max_len = np.argmax(lengths)
        t = ts[index_max_len]

        interpolated_results = []
        for inter in interpolation:
            interpolated_results.append(inter(t))

        interpolated_results = np.array(interpolated_results)

        return t, interpolated_results




    def store_results(self, results):
        features_0d, features_1d, features_2d = self.parallel.sort_features(results[0])

        self.data.features_0d = features_0d
        self.data.features_1d = features_1d
        self.data.features_2d = features_2d

        if self.is_adaptive(results) and not self.model.adaptive:
            # TODO if the model is adaptive perform the complete interpolation here instead.
            raise ValueError("The number of simulation points varies between simulations."
                             + " Try setting adaptive=True in model()")


        for feature in self.data.features_2d:
            if "interpolation" in results[0][feature]:
                raise NotImplementedError("Feature: {feature},".format(feature=feature)
                                          + " no support for >= 2D interpolation")

            else:
                self.data.t[feature] = results[0][feature]["t"]
                self.data.U[feature] = []
                for solved in results:
                    self.data.U[feature].append(solved[feature]["U"])


        for feature in self.data.features_1d:
            if "interpolation" in results[0][feature]:
                ts = []
                interpolations = []
                for solved in results:
                    if "t" in solved[feature]:
                        ts.append(solved[feature]["t"])
                    else:
                        ts.append(solved[self.model.name]["t"])

                    interpolations.append(solved[feature]["interpolation"])

                self.data.t[feature], self.data.U[feature] = self.perform_interpolation(ts, interpolations)
            else:
                # if "t" in results[0][feature]:
                #     self.data.t[feature] = results[0][feature]["t"]
                # else:
                #     self.data.t[feature] = results[0][self.model.name]["t"]
                self.data.t[feature] = results[0][feature]["t"]
                self.data.U[feature] = []
                for solved in results:
                    self.data.U[feature].append(solved[feature]["U"])

                # self.data.U[feature] = np.array(self.data.U[feature])


        for feature in self.data.features_0d:
            self.data.t[feature] = results[0][feature]["t"]
            self.data.U[feature] = []
            for solved in results:
                self.data.U[feature].append(solved[feature]["U"])

        # TODO is this necessary to ensure all results are arrays?
        for feature in self.data.feature_list:
            self.data.U[feature] = np.array(self.data.U[feature])


        self.data.remove_only_invalid_results()




    def evaluate_nodes(self, nodes):
        if self.supress_model_graphics:
            vdisplay = Xvfb()
            vdisplay.start()

        results = []
        pool = mp.Pool(processes=self.CPUs)

        model_parameters = self.create_model_parameters(nodes, self.data.uncertain_parameters)
        for result in tqdm(pool.imap(self.parallel.run, model_parameters),
                           desc="Running model",
                           total=len(nodes.T)):

            results.append(result)

        pool.close()

        if self.supress_model_graphics:
            vdisplay.stop()

        return np.array(results)


    def create_model_parameters(self, nodes, uncertain_parameters):
        model_parameters = []
        for node in nodes.T:
            if isinstance(node, float) or isinstance(node, int):
                node = [node]

            # New setparameters
            parameters = {}
            for j, parameter in enumerate(uncertain_parameters):
                parameters[parameter] = node[j]

            for parameter in self.parameters:
                if parameter.name not in parameters:
                    parameters[parameter.name] = parameter.value

            model_parameters.append(parameters)

        return model_parameters


    # TODO should this check one specific feature.
    # Return false for all features?
    def is_adaptive(self, results):
        """
Test if results is an adaptive result
        """
        for feature in self.data.features_1d + self.data.features_2d:
            u_prev = results[0][feature]["U"]
            for solve in results[1:]:
                u = solve[feature]["U"]
                if u_prev.shape != u.shape:
                    return True
                u_prev = u
        return False



    def run(self, nodes, uncertain_parameters):

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        self.data.uncertain_parameters = uncertain_parameters

        results = self.evaluate_nodes(nodes)

        self.store_results(results)

        return self.data
