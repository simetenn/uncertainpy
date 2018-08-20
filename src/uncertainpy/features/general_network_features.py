from __future__ import absolute_import, division, print_function, unicode_literals

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
    Class for creating NEO spiketrains from a list of spiketrains, for network
    models. The model must return the simulation end time and a list of
    spiketrains.

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
       interpolate : {None, "all", str, list of feature names}, optional
        Which features are irregular, meaning they have a varying number of
        time points between evaluations. An interpolation is performed on
        each irregular feature to create regular results.
        If ``"all"``, all features are interpolated.
        If None, or an empty list, no features are interpolated.
        If str, only that feature is interpolated.
        If list of feature names, all listed features are interpolated.
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
    logger_level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging is performed.
        Default logger level is "info".

    Attributes
    ----------
    features_to_run : list
        Which features to calculate uncertainties for.
    interpolate : list
        A list of irregular features.
    utility_methods : list
        A list of all utility methods implemented. All methods in this class
        that is not in the list of utility methods is considered to be a feature.
    labels : dictionary
        Labels for the axes of each feature, used when plotting.

    Notes
    -----
    All features in this set of features take the following input arguments:

    simulation_end : float
        The simulation end time
    neo_spiketrains : list
        A list of Neo spiketrains.

    The model must return:

    simulation_end : float
        The simulation end time
    spiketrains : list
        A list of spiketrains, each spiketrain is a list of the times when
        a given neuron spikes.

    Raises
    ------
    ImportError
        If neo or quantities is not installed.

    See also
    --------
    GeneralNetworkFeatures.preprocess
    GeneralNetworkFeatures.reference_feature : reference_feature showing the requirements of a feature function.
    """
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 interpolate=None,
                 labels={},
                 units=None,
                 logger_level="info"):

        if not prerequisites:
            raise ImportError("Network features require: neo, quantities")

        super(GeneralNetworkFeatures, self).__init__(new_features=new_features,
                                                     features_to_run=features_to_run,
                                                     interpolate=interpolate,
                                                     labels=labels,
                                                     logger_level=logger_level)
        if units is None:
            self.units = pq.ms
        else:
            self.units = units



    def preprocess(self, simulation_end, spiketrains):
        """
        Preprossesing of the simulation end time `simulation_end` and
        spiketrains `spiketrains` from the model, before the features are
        calculated.

        Parameters
        ----------
        simulation_end : float
            The simulation end time
        spiketrains : list
            A list of spiketrains, each spiketrain is a list of the times when
            a given neuron spikes.

        Returns
        -------
        simulation_end : float
            The simulation end time
        neo_spiketrains : list
            A list of Neo spiketrains.

        Raises
        ------
        ValueError
            If `simulation_end` is np.nan or None.

        Notes
        -----
        This preprocessing makes it so all features get the input
        `simulation_end` and `spiketrains`.

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