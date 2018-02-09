.. _efel:

EfelFeatures
============


An extensive set of features for single neuron voltage traces is
found in the `Electrophys Feature Extraction Library (eFEL)`_.
Uncertainpy has all features in the eFEL library
contained in the :py:class:`~uncertainpy.features.EfelFeatures` class.
As with :ref:`SpikingFeatures <spiking>`,
many of the eFEL features require the start time and end time of the stimulus,
which must be returned as ``info["stimulus_start"]``
and ``info["stimulus_start"]`` in the model function.
eFEL currently contains 153 different features, we briefly list
them here, but refer to  the `eFEL documentation`_ for the definitions of each feature.


.. _Electrophys Feature Extraction Library (eFEL): https://github.com/BlueBrain/eFEL
.. _eFEL documentation: http://efel.readthedocs.io




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


API Reference
-------------

.. autoclass:: uncertainpy.features.EfelFeatures
   :members:
   :inherited-members: