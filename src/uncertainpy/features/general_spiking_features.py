from .spikes import Spikes
from .features import Features


class GeneralSpikingFeatures(Features):
    """
    Class for calculating spikes of a model, works with single neuron models and
    voltage traces.

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
    threshold : {float, int, "auto"}, optional
        The threshold where the model result is considered to have a spike.
        If "auto" the threshold is set to the standard variation of the
        result. Default is -30.
    extended_spikes : bool, optional
        If the found spikes should be extended further out than the threshold
        cuttoff. If True the spikes is considered to start and end where the
        derivative equals 0.5. Default is False.
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
    spikes : Spikes object
        A Spikes object that contain all spikes.
    threshold : {float, int}
        The threshold where the model result is considered to have a spike.
    extended_spikes : bool
        If the found spikes should be extended further out than the threshold
        cuttoff.
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
    uncertainpy.features.Spikes : Class for finding spikes in the model result.
    """
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 adaptive=None,
                 threshold=-30,
                 extended_spikes=False,
                 labels={},
                 verbose_level="info",
                 verbose_filename=None):

        new_utility_methods = ["calculate_spikes"]

        super(GeneralSpikingFeatures, self).__init__(new_features=new_features,
                                                     features_to_run=features_to_run,
                                                     adaptive=adaptive,
                                                     new_utility_methods=new_utility_methods,
                                                     labels=labels,
                                                     verbose_level=verbose_level,
                                                     verbose_filename=verbose_filename)

        self.spikes = None

        self.threshold = threshold
        self.extended_spikes = extended_spikes



    def preprocess(self, time, values, info):
        """
        Calculating spikes from the model result.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        values : array_like
            Result of the model.
        info : dictionary
            A dictionary with info["stimulus_start"] and info["stimulus_end"].

        Returns
        -------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it returns None or numpy.nan.
        values : Spikes
            The spikes found in the model results.
        info : dictionary
            A dictionary with info["stimulus_start"] and info["stimulus_end"].

        Notes
        -----
        Also sets self.values = values, so features have access to self.values if necessary.

        See also
        --------
        uncertainpy.models.Model.run : The model run method
        uncertainpy.features.Spikes : Class for finding spikes in the model result.
        """
        self.values = values

        self.spikes = self.calculate_spikes(time, values, threshold=self.threshold, extended_spikes=self.extended_spikes)

        return time, self.spikes, info


    def calculate_spikes(self, time, values, threshold=-30, extended_spikes=False):
        """
        Calculating spikes of a model result, works with single neuron models and
        voltage traces.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        values : array_like
            Result of the model.
        threshold : {float, int, "auto"}, optional
            The threshold where the model result is considered to have a spike.
            If "auto" the threshold is set to the standard variation of the
            result. Default is -30.
        extended_spikes : bool, optional
            If the found spikes should be extended further out than the threshold
            cuttoff. If True the spikes is considered to start and end where the
            derivative equals 0.5. Default is False.

        Returns
        ------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it returns None or numpy.nan.
        values : Spikes
            The spikes found in the model results.

        See also
        --------
        uncertainpy.features.Features.reference_feature : reference_feature showing the requirements of a feature function.
        uncertainpy.features.Spikes : Class for finding spikes in the model result.
        """
        spikes = Spikes()
        spikes.find_spikes(time, values, threshold=threshold, extended_spikes=extended_spikes)

        return spikes


    def reference_feature(self, time, spikes, info):
        """
        An example of an GeneralSpikingFeature. The feature functions have the
        following requirements, and the input arguments must either be
        returned by ``Model.run`` or ``SpikingFeatures.preprocess``.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            A dictionary with info["stimulus_start"] and
            info["stimulus_end"] set.

        Returns
        -------
        time : {None, numpy.nan, array_like}
            Time values, or equivalent, of the feature, if no time values
            return None or numpy.nan.
        values : array_like
            The feature results, `values`. Returns None if there are no feature
            results and that evaluation are disregarded.

        See also
        --------
        uncertainpy.features.GeneralSpikingFeatures.preprocess : The GeneralSpikingFeatures preprocess method.
        uncertainpy.models.Model.run : The model run method
        """

        # Perform feature calculations here
        time = None
        values = None

        return time, values