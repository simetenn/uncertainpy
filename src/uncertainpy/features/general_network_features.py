import numpy as np

try:
    import neo.core
    import quantities as pq

    prerequisites = True
except ImportError:
    prerequisites = False

from .features import Features


class GeneralNetworkFeatures(Features):
    """
    Class for creating neo spiketrains from spike trains lists, for network
    models.

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
    units : {None, Quantities unit}, optional
        The Quantities unit of the time in the model. If None, ms is used.
        The default is None.
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
    GeneralNetworkFeatures.reference_feature : reference_feature showing the requirements of a feature function.
    """
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 adaptive=None,
                 labels={},
                 units=None,
                 verbose_level="info",
                 verbose_filename=None):

        if not prerequisites:
            raise ImportError("Network features require: neo, quantities")

        super(GeneralNetworkFeatures, self).__init__(new_features=new_features,
                                                     features_to_run=features_to_run,
                                                     adaptive=adaptive,
                                                     labels=labels,
                                                     verbose_level=verbose_level,
                                                     verbose_filename=verbose_filename)
        if units is None:
            self.units = pq.ms
        else:
            self.units = units



    def preprocess(self, simulation_end, spiketrains):
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

        Raises
        ------
        ValueError
            If `simulation_end` is NaN or None.

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


        if simulation_end is None or np.isnan(simulation_end):
            raise ValueError("simulation_end is NaN or None. simulation_end must be the time when the simulation ends.")

        neo_spiketrains = []
        for spiketrain in spiketrains:
            neo_spiketrain = neo.core.SpikeTrain(spiketrain, t_stop=simulation_end, units=self.units)
            neo_spiketrains.append(neo_spiketrain)

        return simulation_end, neo_spiketrains



    def reference_feature(self, simulation_end, neo_spiketrains):
        """
        An example of an GeneralNetworkFeature. The feature functions have the
        following requirements, and the given parameters must either be
        returned by ``model.run`` or ``features.preprocess``.

        Parameters
        ----------
        simulation_end : float
            The simulation end time
        neo_spiketrains : list
            A list of Neo spiketrains.

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