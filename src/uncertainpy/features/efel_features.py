try:
    import efel

    prerequisites = True
except ImportError:
    prerequisites = False

from .features import Features

class EfelFeatures(Features):
    """
    Class for calculating the mean value of each feature in the Efel feature
    library, see: https://github.com/BlueBrain/eFEL.

    Parameters
    ----------
    new_features : {None, callable, list of callables}
        The new features to add. The feature functions have the requirements
        stated in ``example_feature``. If None, no features are added.
        Default is None.
    features_to_run : {"all", None, str, list of feature names}, optional
        Which features to calculate uncertainties for.
        If ``"all"``, the uncertainties will be calculated for all
        implemented and assigned features.
        If None, or an empty list ``[]``, no features will be
        calculated.
        If str, only that feature is calculated.
        If list of feature names, all the listed features will be
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

    strict : bool, optional
        If True missing ``"stimulus_start"`` and ``"stimulus_end"`` from `info`
        raises a RuntimeError. If False the simulation start time is used
        as ``"stimulus_start"`` and the simulation end time is used for
        ``"stimulus_end"``. The decay_time_constant_after_stim feature becomes
        disabled with False. Default is True
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
    logger : logging.Logger object
        Logger object responsible for logging to screen or file.

    Raises
    ------
    RuntimeError
        If strict is True and ``"stimulus_start"`` and ``"stimulus_end"`` are
        missing from `info`.

    Notes
    -----
    Efel features take the parameters ``(t, U, info)`` and require
    info["stimulus_start"] and info["stimulus_end"] to be set.

    Implemented Efel features are:

    ===============================  ===============================  ===============================
    AHP1_depth_from_peak             AHP2_depth_from_peak             AHP_depth
    AHP_depth_abs                    AHP_depth_abs_slow               AHP_depth_diff
    AHP_depth_from_peak              AHP_slow_time                    AHP_time_from_peak
    AP1_amp                          AP1_begin_voltage                AP1_begin_width
    AP1_peak                         AP1_width                        AP2_AP1_begin_width_diff
    AP2_AP1_diff                     AP2_AP1_peak_diff                AP2_amp
    AP2_begin_voltage                AP2_begin_width                  AP2_peak
    AP2_width                        AP_amplitude                     AP_amplitude_change
    AP_amplitude_diff                AP_amplitude_from_voltagebase    AP_begin_indices
    AP_begin_time                    AP_begin_voltage                 AP_begin_width
    AP_duration                      AP_duration_change               AP_duration_half_width
    AP_duration_half_width_change    AP_end_indices                   AP_fall_indices
    AP_fall_rate                     AP_fall_rate_change              AP_fall_time
    AP_height                        AP_phaseslope                    AP_phaseslope_AIS
    AP_rise_indices                  AP_rise_rate                     AP_rise_rate_change
    AP_rise_time                     AP_width                         APlast_amp
    BAC_maximum_voltage              BAC_width                        BPAPAmplitudeLoc1
    BPAPAmplitudeLoc2                BPAPHeightLoc1                   BPAPHeightLoc2
    BPAPatt2                         BPAPatt3                         E10
    E11                              E12                              E13
    E14                              E15                              E16
    E17                              E18                              E19
    E2                               E20                              E21
    E22                              E23                              E24
    E25                              E26                              E27
    E3                               E39                              E39_cod
    E4                               E40                              E5
    E6                               E7                               E8
    E9                               ISI_CV                           ISI_log_slope
    ISI_log_slope_skip               ISI_semilog_slope                ISI_values
    Spikecount                       Spikecount_stimint               adaptation_index
    adaptation_index2                all_ISI_values                   amp_drop_first_last
    amp_drop_first_second            amp_drop_second_last             burst_ISI_indices
    burst_mean_freq                  burst_number                     check_AISInitiation
    decay_time_constant_after_stim   depolarized_base                 doublet_ISI
    fast_AHP                         fast_AHP_change                  interburst_voltage
    inv_fifth_ISI                    inv_first_ISI                    inv_fourth_ISI
    inv_last_ISI                     inv_second_ISI                   inv_third_ISI
    inv_time_to_first_spike          irregularity_index               is_not_stuck
    max_amp_difference               maximum_voltage                  maximum_voltage_from_voltagebase
    mean_AP_amplitude                mean_frequency                   min_AHP_indices
    min_AHP_values                   min_voltage_between_spikes       minimum_voltage
    number_initial_spikes            ohmic_input_resistance           ohmic_input_resistance_vb_ssse
    peak_indices                     peak_time                        peak_voltage
    sag_amplitude                    sag_ratio1                       sag_ratio2
    single_burst_ratio               spike_half_width                 spike_width2
    steady_state_hyper               steady_state_voltage             steady_state_voltage_stimend
    time_constant                    time_to_first_spike              time_to_last_spike
    time_to_second_spike             trace_check                      voltage
    voltage_after_stim               voltage_base                     voltage_deflection
    ===============================  ===============================  ===============================

    See also
    --------
    https://github.com/BlueBrain/eFEL : Documentation for Efel features
    uncertainpy.features.EfelFeatures.example_feature : example_feature showing the requirements of a Efel feature function.
    """
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 adaptive=None,
                 labels={},
                 strict=True,
                 verbose_level="info",
                 verbose_filename=None):

        if not prerequisites:
            raise ImportError("Efel features require: efel")

        efel.reset()

        # TODO implement efel labels
        # implemented_labels = {"nr_spikes": ["number of spikes"],
        #                       "spike_rate": ["spike rate [Hz]"],
        #                       "time_before_first_spike": ["time [ms]"],
        #                       "accommodation_index": ["accommodation index"],
        #                       "average_AP_overshoot": ["voltage [mV]"],
        #                       "average_AHP_depth": ["voltage [mV]"],
        #                       "average_AP_width": ["time [ms]"]
        #                      }
        implemented_labels = {}

        def efel_wrapper(feature_name):
            def feature_function(t, U, info):
                disable = False

                if "stimulus_start" not in info:
                    if strict:
                        raise RuntimeError("Efel features require info['stimulus_start']. "
                                           "No 'stimulus_start' found in info, "
                                           "Set 'stimulus_start', or set strict to "
                                           "False to use initial time as stimulus start")
                    else:
                        info["stimulus_start"] = t[0]
                        self.logger.warning("Efel features require info['stimulus_start']. "
                                            "No 'stimulus_start' found in info, "
                                            "setting stimulus start as initial time")

                if "stimulus_end" not in info:
                    if strict:
                        raise RuntimeError("Efel features require info[stimulus_end]. "
                                           "No 'stimulus_end' found in info, "
                                           "Set 'stimulus_start', or set strict to "
                                           "False to use end time as stimulus end")
                    else:
                        info["stimulus_end"] = t[-1]
                        self.logger.warning("Efel features require info['stimulus_start']. "
                                            "No 'stimulus_end' found in info, "
                                            "setting stimulus end as end time")

                trace = {}
                trace["T"] = t
                trace["V"] = U
                trace["stim_start"] = [info["stimulus_start"]]
                trace["stim_end"] = [info["stimulus_end"]]


                # Disable decay_time_constant_after_stim if no time points left
                # in simulation after stimulation has ended.
                # Otherwise it thros an error
                if feature_name == "decay_time_constant_after_stim":
                    if info["stimulus_end"] >= t[-1]:
                        return None, None

                result = efel.getMeanFeatureValues([trace], [feature_name], raise_warnings=False)

                return None, result[0][feature_name]

            feature_function.__name__ = feature_name
            return feature_function

        super(EfelFeatures, self).__init__(new_features=new_features,
                                           features_to_run=features_to_run,
                                           adaptive=adaptive,
                                           new_utility_methods=[],
                                           labels=implemented_labels,
                                           verbose_level=verbose_level,
                                           verbose_filename=verbose_filename)

        for feature_name in efel.getFeatureNames():
            self.add_features(efel_wrapper(feature_name))

        self.labels = labels
        self.features_to_run = features_to_run





    def example_feature(self, t, U, info):
        """
        An example of an Efel feature. Efel feature functions have the following
        requirements, and the given parameters should either be returned by
        ``model.run`` or ``model.postprocess``.

        Parameters
        ----------
        t : {None, numpy.nan, array_like}
            Time values of the model. If no time values it is None or numpy.nan.
        U : array_like
            Result of the model.
        info : dictionary
            A dictionary with info["stimulus_start"] and info["stimulus_end"]
            set.

        Returns
        -------
        t : None
            No mean Efel feature has time values, so None is returned instead.
        U : array_like
            The feature results, `U`. Returns None if there are no feature
            results and that evaluation will be disregarded.

        See also
        --------
        uncertainpy.features.Features.preprocess : The features preprocess method.
        uncertainpy.models.Model.run : The model run method
        uncertainpy.models.Model.postprocess : The postprocessing method.
        """
        # Perform feature calculations here
        t = None
        U = None

        return t, U