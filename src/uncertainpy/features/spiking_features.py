try:
    import scipy.interpolate
    import scipy.optimize

    prerequisites = True
except ImportError:
    prerequisites = False

from .general_spiking_features import GeneralSpikingFeatures

class SpikingFeatures(GeneralSpikingFeatures):
    """
    Spiking features of a model result, works with single neuron models and
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

    strict : bool, optional
        If True, missing ``"stimulus_start"`` and ``"stimulus_end"`` from `info`
        raises a ValueError. If False the simulation start time is used
        as ``"stimulus_start"`` and the simulation end time is used for
        ``"stimulus_end"``. Default is True.
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
    spikes : Spikes
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
    strict : bool
        If missing info values should raise an error.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.

    Notes
    -----
    The implemented features are:

    ==========================  ==========================
    nr_spikes                   time_before_first_spike
    spike_rate                  average_AP_overshoot
    average_AHP_depth           average_AP_width
    accommodation_index
    ==========================  ==========================

    The features are from:
    Druckmann, S., Banitt, Y., Gidon, A. A., Schurmann, F., Markram, H., and Segev, I.
    (2007). A novel multiple objective optimization framework for constraining conductance-
    based neuron models by experimental data. Frontiers in Neuroscience 1, 7-18. doi:10.
    3389/neuro.01.1.1.001.2007

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
                 strict=True,
                 verbose_level="info",
                 verbose_filename=None):

        if not prerequisites:
            raise ImportError("Spiking features require: scipy")

        implemented_labels = {"nr_spikes": ["Number of spikes"],
                              "spike_rate": ["Spike rate (Hz)"],
                              "time_before_first_spike": ["Time (ms)"],
                              "accommodation_index": ["Accommodation index"],
                              "average_AP_overshoot": ["Voltage (mV)"],
                              "average_AHP_depth": ["Voltage (mV)"],
                              "average_AP_width": ["Time (ms)"]
                             }

        super(SpikingFeatures, self).__init__(new_features=new_features,
                                              features_to_run=features_to_run,
                                              adaptive=adaptive,
                                              threshold=threshold,
                                              extended_spikes=extended_spikes,
                                              labels=implemented_labels,
                                              verbose_level=verbose_level,
                                              verbose_filename=verbose_filename)
        self.labels = labels
        self.strict = strict


    def nr_spikes(self, time, spikes, info):
        """
        The number of spikes in the model result during the stimulus period.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            If ``strict=True``, requires ``info["stimulus_start"]`` and
            ``info['stimulus_end']`` set.

        Returns
        -------
        time : None
        nr_spikes : int
            The number of spikes in the model result.

        Raises
        ------
        ValueError
            If strict is True and ``"stimulus_start"`` and ``"stimulus_end"`` are
            missing from `info`.
        ValueError
            If stimulus_start >= stimulus_end.
        """

        if "stimulus_start" not in info:
            if self.strict:
                raise ValueError("spike_rate require info['stimulus_start']. "
                                 "No 'stimulus_start' found in info, "
                                 "Set 'stimulus_start', or set strict to "
                                 "False to use initial time as stimulus start")
            else:
                info["stimulus_start"] = time[0]
                self.logger.warning("spike_rate features require info['stimulus_start']. "
                                    "No 'stimulus_start' found in info, "
                                    "setting stimulus start as initial time")


        if "stimulus_end" not in info:
            if self.strict:
                raise ValueError("spike_rate require info['stimulus_end']. "
                                 "No 'stimulus_end' found in info, "
                                 "Set 'stimulus_start', or set strict to "
                                 "False to use end time as stimulus end")
            else:
                info["stimulus_end"] = time[-1]
                self.logger.warning("spike_rate require info['stimulus_start']. "
                                    "No 'stimulus_end' found in info, "
                                    "setting stimulus end as end time")

        if info["stimulus_start"] >= info["stimulus_end"]:
            raise ValueError("stimulus_start >= stimulus_end.")

        nr_spikes = 0
        for spike in spikes:
            if info["stimulus_start"] < spike.time_spike < info["stimulus_end"]:
                nr_spikes += 1

        return None, nr_spikes


    def time_before_first_spike(self, time, spikes, info):
        """
        The time from the stimulus start to the first spike occurs.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            If ``strict=True``, requires ``info["stimulus_start"]`` set.

        Returns
        -------
        time : None
        time_before_first_spike : {float, None}
            The time from the stimulus start to the first spike occurs. Returns
            None if there are no spikes on the model result.

        Raises
        ------
        ValueError
            If strict is True and ``"stimulus_start"`` and ``"stimulus_end"`` are
            missing from `info`.
        """

        if "stimulus_start" not in info:
            if self.strict:
                raise ValueError("time_before_first_spike require info['stimulus_start']. "
                                 "No 'stimulus_start' found in info, "
                                 "Set 'stimulus_start', or set strict to "
                                 "False to use initial time as stimulus start")
            else:
                info["stimulus_start"] = time[0]
                self.logger.warning("time_before_first_spike features require info['stimulus_start']. "
                                    "No 'stimulus_start' found in info, "
                                    "setting stimulus start as initial time")


        if spikes.nr_spikes <= 0:
            return None, None

        time = spikes.spikes[0].time_spike - info["stimulus_start"]

        return None, time


    def spike_rate(self, time, spikes, info):
        """
        The spike rate of the model result.

        Number of spikes divided by the duration.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            If ``strict=True``, requires ``info["stimulus_start"]`` and
            ``info['stimulus_end']`` set.

        Returns
        -------
        time : None
        spike_rate : float
            The spike rate of the model result.

        Raises
        ------
        ValueError
            If strict is True and ``"stimulus_start"`` and ``"stimulus_end"`` are
            missing from `info`.
        ValueError
            If stimulus_start >= stimulus_end.
        """

        if "stimulus_start" not in info:
            if self.strict:
                raise ValueError("spike_rate require info['stimulus_start']. "
                                 "No 'stimulus_start' found in info, "
                                 "Set 'stimulus_start', or set strict to "
                                 "False to use initial time as stimulus start")
            else:
                info["stimulus_start"] = time[0]
                self.logger.warning("spike_rate features require info['stimulus_start']. "
                                    "No 'stimulus_start' found in info, "
                                    "setting stimulus start as initial time")


        if "stimulus_end" not in info:
            if self.strict:
                raise ValueError("spike_rate require info['stimulus_end']. "
                                 "No 'stimulus_end' found in info, "
                                 "Set 'stimulus_start', or set strict to "
                                 "False to use end time as stimulus end")
            else:
                info["stimulus_end"] = time[-1]
                self.logger.warning("spike_rate require info['stimulus_start']. "
                                    "No 'stimulus_end' found in info, "
                                    "setting stimulus end as end time")

        if info["stimulus_start"] >= info["stimulus_end"]:
            raise ValueError("stimulus_start >= stimulus_end.")

        if spikes.nr_spikes < 0:
            return None, None

        return None, spikes.nr_spikes/float(info["stimulus_end"] - info["stimulus_start"])


    def average_AP_overshoot(self, time, spikes, info):
        """
        The average action potential overshoot,

        The average of the absolute peak voltage values of all spikes
        (action potentials).

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            Not used in this feature.

        Returns
        -------
        time : None
        average_AP_overshoot : {float, None}
            The average action potential overshoot. Returns None if there are
            no spikes in the model result.
        """

        if spikes.nr_spikes <= 0:
            return None, None

        sum_AP_overshoot = 0
        for spike in spikes:
            sum_AP_overshoot += spike.V_spike

        return None, sum_AP_overshoot/float(spikes.nr_spikes)


    def average_AHP_depth(self, time, spikes, info):
        """
        The average action potential depth.

        The minimum of the model result between two consecutive spikes (action
        potentials).

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            Not used in this feature.

        Returns
        -------
        time : None
        average_AHP_depth : {float, None}
            The average action potential depth. Returns None if there are
            no spikes in the model result.
        """


        if spikes.nr_spikes <= 0:
            return None, None

        sum_AHP_depth = 0
        for i in range(spikes.nr_spikes - 1):
            sum_AHP_depth += min(self.values[spikes[i].global_index:spikes[i+1].global_index])

        return None, sum_AHP_depth/float(spikes.nr_spikes)


    def average_AP_width(self, time, spikes, info):
        """
        The average action potential width.

        The average of the width of every spike (action potential) at the
        midpoint between the start and maximum of each spike.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            Not used in this feature.

        Returns
        -------
        time : None
        average_AP_width : {float, None}
            The average action potential width. Returns None if there are
            no spikes in the model result.
        """

        if spikes.nr_spikes <= 0:
            return None, None

        sum_AP_width = 0
        for spike in spikes:
            V_width = (spike.V_spike + spike.V[0])/2.

            V_interpolation = scipy.interpolate.interp1d(spike.time, spike.V - V_width)

            # root1 = scipy.optimize.fsolve(U_interpolation, (spike.t_spike - spike.t[0])/2. + spike.t[0])
            # root2 = scipy.optimize.fsolve(U_interpolation, (spike.t[-1] - spike.t_spike)/2. + spike.t_spike)

            root1 = scipy.optimize.brentq(V_interpolation, spike.time[0], spike.time_spike)
            root2 = scipy.optimize.brentq(V_interpolation, spike.time_spike, spike.time[-1])

            sum_AP_width += abs(root2 - root1)

        return None, sum_AP_width/float(spikes.nr_spikes)


    def accommodation_index(self, time, spikes, info):
        r"""
        The accommodation index.

        The accommodation index is the average of the difference in length of
        two consecutive interspike intervals normalized by the summed duration
        of the two interspike intervals.

        Parameters
        ----------
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        spikes : Spikes
            Spikes found in the model result.
        info : dictionary
            Not used in this feature.

        Returns
        -------
        time : None
        accommodation_index : {float, None}
            The accommodation index. Returns None if there are
            less than two spikes in the model result.

        Notes
        -----
        The accommodation index is defined as:

        .. math::

            A = \frac{1}{N-k-1} \sum_{i=k}^N \frac{\text{ISI}_i - \text{ISI}_{i-1}}{\text{ISI}_i + \text{ISI}_{i-1}},

        where ISI is the interspike interval, N the number of spikes, and
        k is defined as:

        .. math::

            k = \min \left\{4, \frac{\text{Number of ISIs}}{5}\right\}.
        """

        N = spikes.nr_spikes
        if N <= 1:
            return None, None

        k = min(4, int(round(N-1)/5.))

        ISIs = []
        for i in range(N-1):
            ISIs.append(spikes[i+1].time_spike - spikes[i].time_spike)

        A = 0
        for i in range(k+1, N-1):
            A += (ISIs[i] - ISIs[i-1])/(ISIs[i] + ISIs[i-1])

        return None, A/(N - k - 1)
