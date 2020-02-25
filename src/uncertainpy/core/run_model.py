from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
import six

try:
    from itertools import imap
except ImportError:
    imap = map

from tqdm import tqdm
import numpy as np

try:
    from xvfbwrapper import Xvfb

    prerequisites = True
except ImportError:
    prerequisites = False

from ..data import Data
from ..utils.utility import lengths, contains_nan
from ..utils.logger import get_logger
from .base import ParameterBase
from .parallel import Parallel



class RunModel(ParameterBase):
    """
    Calculate model and feature results for a series of different model parameters,
    and store them in a Data object.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on. For requirements see
        Model.run.
        Default is None.
    parameters: {dict {name: parameter_object}, dict of {name: value or Chaospy distribution}, ...], list of Parameter instances, list [[name, value or Chaospy distribution], ...], list [[name, value, Chaospy distribution or callable that returns a Chaospy distribution],...],}
        List or dictionary of the parameters that should be created.
        On the form ``parameters =``

            * ``{name_1: parameter_object_1, name: parameter_object_2, ...}``
            * ``{name_1:  value_1 or Chaospy distribution, name_2:  value_2 or Chaospy distribution, ...}``
            * ``[parameter_object_1, parameter_object_2, ...]``,
            * ``[[name_1, value_1 or Chaospy distribution], ...]``.
            * ``[[name_1, value_1, Chaospy distribution or callable that returns a Chaospy distribution], ...]``

    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    logger_level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging to file is performed.
        Default logger level is "info".
    CPUs : {int, None, "max"}, optional
        The number of CPUs to use when calculating the model and features.
        If None, no multiprocessing is used.
        If "max", the maximum number of CPUs on the computer
        (multiprocess.cpu_count()) is used.
        Default is "max".


    Attributes
    ----------
    model : uncertainpy.Model or subclass of uncertainpy.Model
        The model to perform uncertainty quantification on.
    parameters : uncertainpy.Parameters
        The uncertain parameters.
    features : uncertainpy.Features or subclass of uncertainpy.Features
        The features of the model to perform uncertainty quantification on.
    CPUs : int
        The number of CPUs used when calculating the model and features.

    See Also
    --------
    uncertainpy.features.Features
    uncertainpy.Parameter
    uncertainpy.Parameters
    uncertainpy.models.Model
    uncertainpy.models.Model.run : Requirements for the model run function.
    """

    def __init__(self,
                 model,
                 parameters,
                 features=None,
                 logger_level="info",
                 CPUs="max"):

        if CPUs == "max":
            import multiprocess

            CPUs = multiprocess.cpu_count()


        self._parallel = Parallel(model=model,
                                  features=features,
                                  logger_level=logger_level)

        super(RunModel, self).__init__(model=model,
                                       parameters=parameters,
                                       features=features,
                                       logger_level=logger_level)

        self.CPUs = CPUs


    @ParameterBase.features.setter
    def features(self, new_features):
        ParameterBase.features.fset(self, new_features)

        self._parallel.features = self.features


    @ParameterBase.model.setter
    def model(self, new_model):
        ParameterBase.model.fset(self, new_model)

        self._parallel.model = self.model


    def apply_interpolation(self, results, feature):
        """
        Perform interpolation of one model/feature using the interpolation
        objects created by Parallel.

        Parameters
        ----------
        results : list
            A list where each element is a result dictionary for each set
            of model evaluations.
            An example:

            .. code-block:: Python

                result = {self.model.name: {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                            "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature1d": {"values": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature0d": {"values": 1,
                                        "time": np.nan},
                          "feature2d": {"values": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature_adaptive": {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                               "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                               "interpolation": scipy interpolation object},
                          "feature_invalid": {"values": np.nan,
                                              "time": np.nan}}

                results = [result 1, result 2, ..., result N]

        feature: str
            Name of a feature or the model.

        Returns
        -------
        time : array_like
            The time array with the greatest number of time steps.
        interpolated_results : list
            A list containing all interpolated model/features results.
            Interpolated at the points of the time results with the greatest
            number of time steps.

        Notes
        -----
        Chooses the time array with the highest number of time points and use
        this time array to interpolate the model/feature results in each of
        those points. If an interpolation is None, gives numpy.nan instead.
        """
        logger = get_logger(self)

        time_lengths = []
        for result in results:
            time_lengths.append(len(result[feature]["time"]))

        index_max_len = np.argmax(time_lengths)
        time = results[index_max_len][feature]["time"]

        interpolated_results = []
        for result in results:
            interpolation = result[feature]["interpolation"]

            if interpolation is None:
                interpolated_results.append(np.nan)
                logger.error("{}: Unknown error while creating the interpolation".format(feature))

            elif isinstance(interpolation, six.string_types):
                interpolated_results.append(np.nan)
                logger.error(interpolation)

            else:
                interpolated_results.append(interpolation(time))

        return time, interpolated_results





    def results_to_data(self, results):
        """
        Store `results` in a Data object.

        Stores the time and (interpolated) results for the model and each
        feature in a Data object. Performs the interpolation calculated in
        Parallel, if the result is irregular.

        Parameters
        ----------
        results : list
            A list where each element is a result dictionary for each set
            of model evaluations.
            An example:

            .. code-block:: Python

                result = {self.model.name: {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                            "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature1d": {"values": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature0d": {"values": 1,
                                        "time": np.nan},
                          "feature2d": {"values": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature_adaptive": {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                               "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                               "interpolation": scipy interpolation object},
                          "feature_invalid": {"values": np.nan,
                                              "time": np.nan}}

                results = [result 1, result 2, ..., result N]

        Returns
        -------
        data : Data object
            A Data object with time and (interpolated) results for the model and
            each feature.

        Notes
        -----
        Sets the following in data, if applicable:
        1. ``data["model/features"].evaluations``, which contains all ``values``
        2. ``data["model/features"].time``
        3. ``data["model/features"].labels``
        4. ``data.model_name``

        See Also
        --------
        uncertainpy.Data
        """
        logger = get_logger(self)

        data = Data(logger_level=self._logger_level)

        # Add features and labels
        for feature in results[0]:
            data.add_features(feature)

            if feature == self.model.name:
                data[feature]["labels"] = self.model.labels
            elif feature in self.features.labels:
                data[feature]["labels"] = self.features.labels[feature]

        data.model_name = self.model.name
        data.model_ignore = self.model.ignore

        def add_results(results, data, feature):
            data[feature].time = []
            data[feature].evaluations = []

            for result in results:
                data[feature].evaluations.append(result[feature]["values"])
                data[feature].time.append(result[feature]["time"])

        # results = self.regularize_nan_results(results)

        # Check if features are irregular without being specified as a interpolate
        # TODO if the feature is irregular, perform the complete interpolation here instead
        # for feature in data:
        #     if (feature == self.model.name and not (self.model.ignore or self.model.interpolate)) \
        #         or (feature != self.model.name and feature not in self.features.interpolate):
        #             if not self.is_regular(results, feature):
        #                 data.error.append(feature)
        #                 data[feature].time = []
        #                 data[feature].evaluations = []

        #                 for result in results:
        #                     data[feature].evaluations.append(result[feature]["values"])
        #                     data[feature].time.append(result[feature]["time"])

                        # raise ValueError("{}: The number of points varies between evaluations.".format(feature)
                        #                  + " Try setting interpolate".format(feature))


        # Store all results in data, interpolate as needed
        # TODO: save raw result instead of interpolated result?
        for feature in data:
            # Interpolate the data if it is irregular, and ignore the model if required
            if feature in self.features.interpolate or \
                    (feature == self.model.name and self.model.interpolate and not self.model.ignore):
                # TODO implement interpolation of >= 2d data, part2
                if np.ndim(results[0][feature]["values"]) >= 2:
                    # raise NotImplementedError("Feature: {feature},".format(feature=feature)
                    #                           + " no support for >= 2D interpolation")
                    logger.error("{feature}:".format(feature=feature)
                                 + " no support for >= 2D interpolation implemented")

                    add_results(results, data, feature)


                elif np.ndim(results[0][feature]["values"]) == 1:
                    data[feature].time, data[feature].evaluations = self.apply_interpolation(results, feature)

                # Interpolating a 0D result makes no sense, so if a 0D feature
                # is supposed to be interpolated store it as normal
                elif np.ndim(results[0][feature]["values"]) == 0:
                    logger.warning("{feature}: ".format(feature=feature) +
                                   "returns a 0D result. No interpolation is performed.")

                    data[feature].time = results[0][feature]["time"]

                    data[feature].evaluations = []
                    for result in results:
                        data[feature].evaluations.append(result[feature]["values"])


            elif feature == self.model.name and self.model.ignore:
                add_results(results, data, feature)

            else:
                # Check if features are irregular without being specified as a interpolate
                # TODO if the feature is irregular, perform the complete interpolation here instead
                if not self.is_regular(results, feature):
                    data.error.append(feature)

                    add_results(results, data, feature)

                    if feature == self.model.name:
                        msg = "{}: The number of points varies between evaluations. ".format(feature) + \
                              "Make sure {} returns the same number of points with different parameters, ".format(feature) + \
                              "implement Model.postprocess, or try to set interpolate=True."
                    else:
                        msg = "{}: The number of points varies between evaluations. ".format(feature) + \
                              "Make sure {} returns the same number of points, ".format(feature) + \
                              "or try add {} to interpolate=[].".format(feature)

                    logger.error(msg)

                    # raise ValueError("{}: The number of points varies between evaluations.".format(feature)
                    #                  + " Try setting interpolate".format(feature))


                else:
                    # Store data from results in a Data object
                    data[feature].time = results[0][feature]["time"]

                    data[feature].evaluations = []
                    for result in results:
                        data[feature].evaluations.append(result[feature]["values"])

        return data




    def evaluate_nodes(self, nodes, uncertain_parameters):
        """
        Evaluate the the model and calculate the features
        for the nodes (values) for the uncertain parameters.

        Parameters
        ----------
        nodes : array
            The values for the uncertain parameters
            to evaluate the model and features for.
        uncertain_parameters : list
            A list of the names of all uncertain parameters.

        Returns
        -------
        results : list
            A list where each element is a result dictionary for each set
            of model evaluations.
            An example:

            .. code-block:: Python

                result = {self.model.name: {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                            "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature1d": {"values": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature0d": {"values": 1,
                                        "time": np.nan},
                          "feature2d": {"values": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature_adaptive": {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                               "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                               "interpolation": scipy interpolation object},
                          "feature_invalid": {"values": np.nan,
                                              "time": np.nan}}

                results = [result 1, result 2, ..., result N]

        Raises
        ------
        ImportError
            If xvfbwrapper is not installed.
        """
        if self.model.suppress_graphics:
            if not prerequisites:
                raise ImportError("Running with suppress_graphics require: xvfbwrapper")

            vdisplay = Xvfb()
            vdisplay.start()

        results = []

        model_parameters = self.create_model_parameters(nodes, uncertain_parameters)

        if self.CPUs:
            import multiprocess as mp

            pool = mp.Pool(processes=self.CPUs)

            # pool.map(self._parallel.run, model_parameters)
            # chunksize = int(np.ceil(len(model_parameters)/self.CPUs))
            chunksize = 1
            for result in tqdm(pool.imap(self._parallel.run, model_parameters, chunksize),
                               desc="Running model",
                               total=len(nodes.T)):

                results.append(result)

            pool.close()

        else:
            for result in tqdm(imap(self._parallel.run, model_parameters),
                               desc="Running model",
                               total=len(nodes.T)):

                results.append(result)



        if self.model.suppress_graphics:
            vdisplay.stop()

        return results



    def create_model_parameters(self, nodes, uncertain_parameters):
        """
        Combine nodes (values) with the uncertain parameter names to create a
        list of dictionaries corresponding to the model values for each
        model evaluation.

        Parameters
        ----------
        nodes : array
            A series of different set of parameters. The model and each feature is
            evaluated for each set of parameters in the series.
        uncertain_parameters : list
            A list of names of the uncertain parameters.

        Returns
        -------
        model_parameters : list
            A list where each element is a dictionary with the model parameters
            for a single evaluation.
            An example:

            .. code-block:: Python

                model_parameter = {"parameter 1": value 1, "parameter 2": value 2, ...}
                model_parameters = [model_parameter 1, model_parameter 2, ...]

        """

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


    def is_regular(self, results, feature):
        """
        Test if `feature` in `results` is regular or not, meaning it has a
        varying number of values for each evaluation. Ignores results that
        contains numpy.nan.

        Parameters
        ----------
        results : list
            A list where each element is a result dictionary for each set
            of model evaluations.
            An example:

            .. code-block:: Python

                result = {self.model.name: {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                            "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature1d": {"values": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature0d": {"values": 1,
                                        "time": np.nan},
                          "feature2d": {"values": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature_adaptive": {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                               "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                               "interpolation": scipy interpolation object},
                          "feature_invalid": {"values": np.nan,
                                              "time": np.nan}}

                results = [result 1, result 2, ..., result N]

        feature: str
            Name of a feature or the model.

        Returns
        -------
        bool
            True if the feature is regular or False if the feature is irregular.
        """
        i = 0
        for result in results:
            i += 1
            if not contains_nan(result[feature]["values"]):
                values_prev = result[feature]["values"]

                # If object array it is not regular
                if isinstance(values_prev, np.ndarray):
                    tmp_array = values_prev
                else:
                    tmp_array = np.array(values_prev)

                if tmp_array.dtype is np.dtype("object"):
                    return False

                break

        for result in results[i:]:
            values = result[feature]["values"]

            # If object array it is not regular
            if isinstance(values, np.ndarray):
                tmp_array = values
            else:
                tmp_array = np.array(values)

            if tmp_array.dtype is np.dtype("object"):
                return False


            if not contains_nan(values):
                try:
                    if np.shape(values_prev) != np.shape(values):
                        return False

                except ValueError:
                    if lengths(values_prev) != lengths(values):
                        return False

                values_prev = values

        return True


    def run(self, nodes, uncertain_parameters):
        """
        Evaluate the the model and calculate the features
        for the nodes (values) for the uncertain parameters.
        The results are interpolated as necessary.

        Parameters
        ----------
        nodes : array
            A series of different set of parameters. The model and each feature is
            evaluated for each set of parameters in the series.
        uncertain_parameters : list
            A list of names of the uncertain parameters.

        Returns
        -------
        data : Data object
            A Data object with time and (interpolated) results for
            the model and each feature.

        See Also
        --------
        uncertainpy.Data
        """

        if isinstance(uncertain_parameters, six.string_types):
            uncertain_parameters = [uncertain_parameters]

        results = self.evaluate_nodes(nodes, uncertain_parameters)

        data = self.results_to_data(results)
        data.uncertain_parameters = uncertain_parameters

        return data

    # Currently not needed
    def regularize_nan_results(self, results):
        """
        Regularize arrays with that only contain numpy.nan values.

        Make each result for each feature have the same the same shape, if they
        only contain numpy.nan values.

        Parameters
        ----------
        results : list
            A list where each element is a result dictionary for each set
            of model evaluations.
            An example:

            .. code-block:: Python

                result = {self.model.name: {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                            "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature1d": {"values": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature0d": {"values": 1,
                                        "time": np.nan},
                          "feature2d": {"values": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature_adaptive": {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                               "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                               "interpolation": scipy interpolation object},
                          "feature_invalid": {"values": np.nan,
                                              "time": np.nan}}

                results = [result 1, result 2, ..., result N]

        Returns
        -------
        results : list
            A list with where the only nan results have been regularized.
            On the form:

            .. code-block:: Python

                result = {self.model.name: {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                            "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature1d": {"values": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature0d": {"values": 1,
                                        "time": np.nan},
                          "feature2d": {"values": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                                        "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                          "feature_adaptive": {"values": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                               "time": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                               "interpolation": scipy interpolation object},
                          "feature_invalid": {"values": np.nan,
                                              "time": np.nan}}

                results = [result 1, result 2, ..., result N]

        """

        warnings.warn(
            "regularize_nan_results is no longer used as nan results no longer are required to be regular.",
            DeprecationWarning
        )

        def regularize(results, data):
            """
            Parameters
            ---------
            data : {"values", "time"}
                Which data to regularize, either time or values
            """
            features = results[0].keys()
            for feature in features:
                # Find shape of the first result that is not only nan values
                shape = np.shape(results[0][feature][data])
                for i in range(len(results)):
                    values = results[i][feature][data]
                    if not np.all(np.isnan(values)):
                        shape = np.shape(values)
                        break

                # Find all results that is only nan, and change their shape if
                # the shape is wrong
                for i in range(len(results)):
                    values = results[i][feature][data]
                    if np.all(np.isnan(values)) and np.shape(values) != shape:
                        results[i][feature][data] = np.full(shape, np.nan, dtype=float)

            return results

        results = regularize(results, "values")
        results = regularize(results, "time")

        return results
