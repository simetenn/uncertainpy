from tqdm import tqdm
from xvfbwrapper import Xvfb

import numpy as np
import multiprocess as mp

from .data import Data
from .parallel import Parallel
from .base import ParameterBase


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
                               "interpolation": <scipy.interpolate.fitpack2. \
                                                InterpolatedUnivariateSpline \
                                                object at 0x7f1c78f0d4d0>},
          "feature_invalid": {"U": np.nan,
                              "t": np.nan}}

results = [result 1, result 2, ..., result N]
"""

class RunModel(ParameterBase):
    """
    Calculate model and feature results for a series if different model parameters,
    and store them in a Data object.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on.
    parameters : {None, Parameters instance, list of Parameter instances, list with [[name, value, distribution], ...]}
        Either None, a Parameters instance or a list the parameters that should be created.
        The two lists are similar to the arguments sent to Parameters.
    features : {None, GeneralFeatures or GeneralFeatures subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if set. Default is None.
    CPUs : int, optional
        The number of CPUs to use when calculating the model and features.
        Default is number of CPUs on the computer (multiprocess.cpu_count()).
    suppress_model_graphics : bool, optional
        Suppress all model graphics created by the model.
        Default is True.


    Attributes
    ----------
    model
    parameters
    features
    logger : logging.Logger object
        Logger object responsible for logging to screen or file.
    CPUs : int
        The number of CPUs used when calculating the model and features.
    suppress_model_graphics : bool
        Suppress all model graphics created by the model.
    """

    def __init__(self,
                 model,
                 parameters,
                 features=None,
                 verbose_level="info",
                 verbose_filename=None,
                 CPUs=mp.cpu_count(),
                 suppress_model_graphics=True):


        self._parallel = Parallel(model=model,
                                 features=features,
                                 verbose_level=verbose_level,
                                 verbose_filename=verbose_filename)

        super(RunModel, self).__init__(model=model,
                                       parameters=parameters,
                                       features=features,
                                       verbose_level=verbose_level,
                                       verbose_filename=verbose_filename)

        self.CPUs = CPUs
        self.suppress_model_graphics = suppress_model_graphics



    @ParameterBase.features.setter
    def features(self, new_features):
        ParameterBase.features.fset(self, new_features)

        self._parallel.features = self.features


    @ParameterBase.model.setter
    def model(self, new_model):
        ParameterBase.model.fset(self, new_model)

        self._parallel.model = self.model


    def apply_interpolation(self, t_interpolate, interpolation):
        lengths = []
        for t_tmp in t_interpolate:
            lengths.append(len(t_tmp))

        index_max_len = np.argmax(lengths)
        t = t_interpolate[index_max_len]

        interpolated_results = []
        for inter in interpolation:
            interpolated_results.append(inter(t))

        interpolated_results = np.array(interpolated_results)

        return t, interpolated_results


    def results_to_data(self, results):
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

        data = Data()

        # Add features and labels
        for feature in results[0]:
            data.add_features(feature)

            if feature == self.model.name:
                data[feature]["labels"] = self.model.labels
            elif feature in self.features.labels:
                data[feature]["labels"] = self.features.labels[feature]

        data.model_name = self.model.name


        # Check if features are adaptive withouth being specified as a adaptive
        # TODO if the feature is adaptive, perform the complete interpolation here instead
        for feature in data:
            if self.is_adaptive(results, feature):
                if (feature == self.model.name and not self.model.adaptive) \
                    or (feature != self.model.name and feature not in self.features.adaptive):
                    raise ValueError("{}: The number of points varies between runs.".format(feature)
                                     + " Try setting adaptive to True in {}".format(feature))


        # Store all results in data, interpolate as needed
        for feature in data:
            # Interpolate the data if it is adaptive
            # if "interpolation" in results[0][feature]:
            if feature in self.features.adaptive or \
                    (feature == self.model.name and self.model.adaptive):
                # TODO implement interpolation of >= 2d data, part2
                if np.ndim(results[0][feature]["U"]) >= 2:
                    raise NotImplementedError("Feature: {feature},".format(feature=feature)
                                              + " no support for >= 2D interpolation")

                elif np.ndim(results[0][feature]["U"]) == 1:
                    t_interpolate = []
                    interpolations = []
                    for result in results:
                        # if "t" in result[feature]:
                        if not np.all(np.isnan(result[feature]["t"])):
                            t_interpolate.append(result[feature]["t"])
                        elif not np.all(np.isnan(result[self.model.name]["t"])):
                            t_interpolate.append(result[self.model.name]["t"])
                        else:
                            raise ValueError("Neither {} or model has t values to use in interpolation".format(feature))

                        interpolations.append(result[feature]["interpolation"])

                    data[feature]["t"], data[feature]["U"] = self.apply_interpolation(t_interpolate, interpolations)

                # Interpolating a 0D result makes no sense, so if a 0D feature
                # is supposed to be interpolated store it as normal
                elif np.ndim(results[0][feature]["U"]) == 0:
                    self.logger.warning("Feature: {feature}, ".format(feature=feature) +
                                        "is a 0D result. No interpolation is performed")

                    t = results[0][feature]["t"]
                    if not np.all(np.isnan(t)):
                        data[feature]["t"] = t

                    data[feature]["U"] = []
                    for result in results:
                        data[feature]["U"].append(result[feature]["U"])

            else:
                # Store data from results in a Data object
                t = results[0][feature]["t"]
                if not np.all(np.isnan(t)):
                    data[feature]["t"] = t

                data[feature]["U"] = []
                for result in results:
                    data[feature]["U"].append(result[feature]["U"])

        # TODO is this necessary to ensure all results are arrays?
        for feature in data:
            if "t" in data[feature]:
                data[feature]["t"] = np.array(data[feature]["t"])
            data[feature]["U"] = np.array(data[feature]["U"])

        data.remove_only_invalid_results()

        return data




    def evaluate_nodes(self, nodes, uncertain_parameters):
        if self.suppress_model_graphics:
            vdisplay = Xvfb()
            vdisplay.start()

        results = []
        pool = mp.Pool(processes=self.CPUs)

        model_parameters = self.create_model_parameters(nodes, uncertain_parameters)
        for result in tqdm(pool.imap(self._parallel.run, model_parameters),
                           desc="Running model",
                           total=len(nodes.T)):

            results.append(result)

        pool.close()

        if self.suppress_model_graphics:
            vdisplay.stop()

        return np.array(results)


    def create_model_parameters(self, nodes, uncertain_parameters):
        model_parameters = []
        for node in nodes.T:
            if node.ndim == 0:
                node = [node]

            # New set parameters
            parameters = {}
            for j, parameter in enumerate(uncertain_parameters):
                parameters[parameter] = node[j]

            for parameter in self.parameters:
                if parameter.name not in parameters:
                    parameters[parameter.name] = parameter.value

            model_parameters.append(parameters)

        return model_parameters


    def is_adaptive(self, results, feature):
        """
Test if results is an adaptive result

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
        u_prev = results[0][feature]["U"]
        for solve in results[1:]:
            u = solve[feature]["U"]

            if not np.all(np.isnan(u)):
                if u_prev.shape != u.shape:
                    return True
                u_prev = u

        return False



    def run(self, nodes, uncertain_parameters):

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        results = self.evaluate_nodes(nodes, uncertain_parameters)

        data = self.results_to_data(results)
        data.uncertain_parameters = uncertain_parameters

        return data
