from ..utils import create_logger

class Features(object):
    """
    Class for calculating features of a model.

    Parameters
    ----------
    new_features : {None, callable, list of callables}
        The new features to add. The feature functions have the requirements
        stated in ``reference_feature``. If None, no features are added.
        Default is None.
    features_to_run : {"all", None, str, list of feature names}, optional
        Which features to calculate uncertainties for.
        If ``"all"``, the uncertainties are calculated for all
        implemented and assigned features.
        If None, or an empty list ``[]``, no features are
        calculated.
        If str, only that feature is calculated.
        If list of feature names, all the listed features are
        calculated. Default is ``"all"``.
    new_utility_methods : {None, list}, optional
        A list of new utility methods. All methods in this class that is not in
        the list of utility methods, is considered to be a feature.
        Default is None.
    adaptive : {None, "all", str, list of feature names}, optional
        Which features that are adaptive, meaning they have a varying number of
        time points between evaluations. An interpolation is performed on
        each adaptive feature to create regular results.
        If ``"all"``, all features are set to adaptive.
        If None, or an empty list, no features are adaptive.
        If str, only that feature is adaptive.
        If list of feature names, all listed are adaptive.
        Default is None.
    labels : dictionary, optional
        A dictionary with key as the feature name and the value as a list of
        labels for each axis. The number of elements in the list corresponds
        to the dimension of the feature. Example:

        .. code-block:: Python

            new_labels = {"0d_feature": ["x-axis"],
                          "1d_feature": ["x-axis", "y-axis"],
                          "2d_feature": ["x-axis", "y-axis", "z-axis"]
                         }

    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if a filename is given.
        Default is None.

    Attributes
    ----------
    features_to_run : list
        Which features to calculate uncertainties for.
    adaptive : list
        A list of the adaptive features.
    utility_methods : list
        A list of all utility methods implemented. All methods in this class
        that is not in the list of utility methods is considered to be a feature.
    labels : dictionary
        Labels for the axes of each feature, used when plotting.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.

    See also
    --------
    uncertainpy.features.Features.reference_feature : reference_feature showing the requirements of a feature function.
    """
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 new_utility_methods=None,
                 adaptive=None,
                 labels={},
                 preprocess=None,
                 verbose_level="info",
                 verbose_filename=None):

        self.utility_methods = ["calculate_feature",
                                "calculate_features",
                                "calculate_all_features",
                                "__init__",
                                "implemented_features",
                                "preprocess",
                                "add_features",
                                "reference_feature",
                                "_preprocess"]

        if new_utility_methods is None:
            new_utility_methods = []

        self._features_to_run = []
        self._adaptive = None
        self._labels = {}

        self.utility_methods += new_utility_methods

        self.adaptive = adaptive

        if new_features is not None:
            self.add_features(new_features, labels=labels)
        if preprocess is not None:
            self.preprocess = preprocess

        self.labels = labels
        self.features_to_run = features_to_run

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


    @property
    def preprocess(self):
        """
        Preprossesing of the time `time` and results `values` from the model, before the
        features are calculated.

        No preprocessing is performed, and the direct model results are
        currently returned.
        If preprocessing is needed it should follow the below format.

        Parameters
        ----------
        *model_results
            Variable length argument list. Is the values that ``model.run()``
            returns. By default it contains `time` and `values`, and then any number of
            optional `info` values.

        Returns
        -------
        preprocess_results
            Returns any number of values that are sent to each feature.
            The values returned must compatible with the input arguments of
            all features.

        Notes
        -----
        Perform a preprossesing of the model results before the results are sent
        to the calculation of each feature. It is used to perform common
        calculations that each feature needs to perform, to reduce the number of
        necessary calculations. The values returned must therefore be compatible
        with the input arguments to each features.


        See also
        --------
        uncertainpy.models.Model.run : The model run method
        """
        return self._preprocess


    def _preprocess(self, *model_result):
        return model_result

    @preprocess.setter
    def preprocess(self, new_preprocess_function):
        if not callable(new_preprocess_function):
            raise TypeError("preprocess function must be callable")

        self._preprocess = new_preprocess_function

    @property
    def labels(self):
        """
        Labels for the axes of each feature, used when plotting.

        Parameters
        ----------
        new_labels : dictionary
            A dictionary with key as the feature name and the value as a list of
            labels for each axis. The number of elements in the list corresponds
            to the dimension of the feature. Example:

            .. code-block:: Python

                new_labels = {"0d_feature": ["x-axis"],
                              "1d_feature": ["x-axis", "y-axis"],
                              "2d_feature": ["x-axis", "y-axis", "z-axis"]
                             }
        """
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        self.labels.update(new_labels)


    @property
    def features_to_run(self):
        """
        Which features to calculate uncertainties for.

        Parameters
        ----------
        new_features_to_run : {"all", None, str, list of feature names}
            Which features to calculate uncertainties for.
            If ``"all"``, the uncertainties are calculated for all
            implemented and assigned features.
            If None, or an empty list , no features are
            calculated.
            If str, only that feature is calculated.
            If list of feature names, all listed features are
            calculated. Default is ``"all"``.

        Returns
        -------
        list
            A list of features to calculate uncertainties for.
        """
        return self._features_to_run

    @features_to_run.setter
    def features_to_run(self, new_features_to_run):
        if new_features_to_run == "all":
            self._features_to_run = self.implemented_features()
        elif new_features_to_run is None:
            self._features_to_run = []
        elif isinstance(new_features_to_run, str):
            self._features_to_run = [new_features_to_run]
        else:
            self._features_to_run = new_features_to_run


    @property
    def adaptive(self):
        """
        Adaptive features.

        Which features that are adaptive, meaning they have a varying number of
        time points between evaluations. An interpolation is performed on
        each adaptive feature to create regular results.

        Parameters
        ----------
        new_features_to_run : {None, "all", str, list of feature names}
            If ``"all"``, all features are set to adaptive.
            If None, or an empty list, no features are adaptive.
            If str, only that feature is adaptive.
            If list of feature names, all listed are adaptive.
            Default is None.

        Returns
        -------
        list
            A list of adaptive features.
        """
        return self._adaptive


    @adaptive.setter
    def adaptive(self, new_adaptive):
        if new_adaptive == "all":
            self._adaptive = self.implemented_features()
        elif new_adaptive is None:
            self._adaptive = []
        elif isinstance(new_adaptive, str):
            self._adaptive = [new_adaptive]
        else:
            self._adaptive = new_adaptive



    def add_features(self, new_features, labels={}):
        """
        Add new features.

        Parameters
        ----------
        new_features : {callable, list of callables}
            The new features to add. The feature functions have the requirements
            stated in ``reference_feature``.
        labels : dictionary, optional
            A dictionary with the labels for the new features. The keys are the
            feature function names and the values are a list of labels for each
            axis. The number of elements in the list corresponds
            to the dimension of the feature. Example:

            .. code-block:: Python

                new_labels = {"0d_feature": ["x-axis"],
                              "1d_feature": ["x-axis", "y-axis"],
                              "2d_feature": ["x-axis", "y-axis", "z-axis"]
                             }

        Raises
        ------
        TypeError
            Raises a TypeError if `new_features`  is not callable or list of
            callables.

        Notes
        -----
        The features added are not added to ``features_to_run``.
        ``features_to_run`` must be set manually afterwards.

        See also
        --------
        uncertainpy.features.Features.reference_feature : reference_feature showing the requirements of a feature function.
        """
        if callable(new_features):
            setattr(self, new_features.__name__, new_features)
            # self.features_to_run.append(new_features.__name__)

            tmp_label = labels.get(new_features.__name__)
            if tmp_label is not None:
                self.labels[new_features.__name__] = tmp_label
        else:
            try:
                for feature in new_features:
                    if callable(feature):
                        setattr(self, feature.__name__, feature)
                        # self.features_to_run.append(feature.__name__)

                        tmp_lables = labels.get(feature.__name__)
                        if tmp_lables is not None:
                            self.labels[feature.__name__] = tmp_lables
                    else:
                        raise TypeError("Feature in iterable is not callable")
            except TypeError as error:
                msg = "Added features must be a callable or list of callables"
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise



    def calculate_feature(self, feature_name, *preprocess_results):
        """
        Calculate feature with `feature_name`.

        Parameters
        ----------
        feature_name : str
            Name of feature to calculate.
        *preprocess_results
            The values returned by ``preprocess``. These values are sent
            as input arguments to each feature. By default preprocess returns
            the values that ``model.run()`` returns, which contains `time` and
            `values`, and then any number of optional `info` values.
            The implemented features require that `info` is a single
            dictionary with the information stored as key-value pairs.
            Certain features require specific keys to be present.

        Returns
        -------
        time : {None, numpy.nan, array_like}
            Time values, or equivalent, of the feature, if no time values
            returns None or numpy.nan.
        values : array_like
            The feature results, `values` must either be regular (have the same
            number of points for different paramaters) or be able to be
            interpolated.

        Raises
        ------
        TypeError
            If `feature_name` is a utility method.

        Notes
        -----
        Checks that the feature returns two arguments.


        See also
        --------
        uncertainpy.models.Model.run : The model run method
        """
        if feature_name in self.utility_methods:
            raise TypeError("{} is a utility method".format(feature_name))


        try:
            feature_result = getattr(self, feature_name)(*preprocess_results)
        except Exception as error:
            msg = "Error in calculation of feature: {}".format(feature_name)
            if not error.args:
                error.args = ("",)
            error.args = error.args + (msg,)
            raise

        # Check that time, and values is returned
        try:
            time_feature, values_feature = feature_result
        except (ValueError, TypeError) as error:
            msg = "feature {} must return time and values (return time, values | return None, U)".format(feature_name)
            if not error.args:
                error.args = ("",)
            error.args = error.args + (msg,)
            raise

        return feature_result



    def calculate_features(self, *preprocess_results):
        """
        Calculate all features in ``features_to_run``.

        Parameters
        ----------
        *preprocess_results
            The values returned by ``preprocess``. These values are sent
            as input arguments to each feature. By default preprocess returns
            the values that ``model.run()`` returns, which contains `time` and
            `values`, and then any number of optional `info` values.
            The implemented features require that `info` is a single
            dictionary with the information stored as key-value pairs.
            Certain features require specific keys to be present.

        Returns
        -------
        results : dictionary
            A dictionary where the keys are the feature names
            and the values are a dictionary with the time values `time` and feature
            results on `values`, on the form ``{"time": time, "values": values}``.

        Raises
        ------
        TypeError
            If `feature_name` is a utility method.

        Notes
        -----
        Checks that the feature returns two values.

        See also
        --------
        uncertainpy.features.Features.calculate_feature : Method for calculating a single feature.
        """
        results = {}
        for feature in self.features_to_run:
            time_feature, values_feature = self.calculate_feature(feature, *preprocess_results)

            results[feature] = {"time": time_feature, "values": values_feature}

        return results


    def calculate_all_features(self, *args):
        """
        Calculate all implemented features.

        Parameters
        ----------
        *preprocess_results
            The values returned by ``preprocess``. These values are sent
            as input arguments to each feature. By default preprocess returns
            the values that ``model.run()`` returns, which contains `time` and
            `values`, and then any number of optional `info` values.
            The implemented features require that `info` is a single
            dictionary with the information stored as key-value pairs.
            Certain features require specific keys to be present.

        Returns
        -------
        results : dictionary
            A dictionary where the keys are the feature names
            and the values are a dictionary with the time values `time` and feature
            results on `values`, on the form ``{"time": t, "values": U}``.

        Raises
        ------
        TypeError
            If `feature_name` is a utility method.

        Notes
        -----
        Checks that the feature returns two values.

        See also
        --------
        uncertainpy.features.Features.calculate_feature : Method for calculating a single feature.
        """
        results = {}
        for feature in self.implemented_features():
            time_feature, values_feature = self.calculate_feature(feature, *args)

            results[feature] = {"time": time_feature, "values": values_feature}

        return results


    def implemented_features(self):
        """
        Return a list of all callable methods in feature, that are not utility
        methods, does not starts with "_" and not a method of a general python object.

        Returns
        -------
        list
            A list of all callable methods in feature, that are not utility
            methods.
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods and method not in dir(object) and not method.startswith("_")]


    def reference_feature(self, *preprocess_results):
        """
        An example feature. Feature function have the following requirements.

        Parameters
        ----------
        *preprocess_results
            Variable length argument list. Is the values that
            ``Features.preprocess`` returns. By default ``Features.preprocess``
            returns the same values as ``Model.run`` returns.

        Returns
        -------
        time : {None, numpy.nan, array_like}
            Time values, or equivalent, of the feature, if no time values
            return None or numpy.nan.
        values : array_like
            The feature results, `values` must either be regular (have the same
            number of points for different paramaters) or be able to be
            interpolated. If there are no feature results return
            None or numpy.nan instead of `values` and that evaluation are
            disregarded.

        See also
        --------
        uncertainpy.features.Features.preprocess : The features preprocess method.
        uncertainpy.models.Model.run : The model run method
        uncertainpy.models.Model.postprocess : The postprocessing method.
        """

        # Perform feature calculations here
        time = None
        values = None

        return time, values