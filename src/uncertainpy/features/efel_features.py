try:
    import efel

    prerequisites = True
except ImportError:
    prerequisites = False

from .features import Features

class EfelFeatures(Features):
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
            def function(t, U, info):
                disable = False

                if "stimulus_start" not in info:
                    if strict:
                        raise RuntimeError("Efel features require info[stimulus_start]. "
                                           "No stimulus_start found in info, "
                                           "Set stimulus_start, or set strict to "
                                           "False to use initial time as stimulus start")
                    else:
                        info["stimulus_start"] = t[0]
                        self.logger.warning("Efel features require info[stimulus_start]. "
                                            "No stimulus_start found in info, "
                                            "setting stimulus start as initial time")

                if "stimulus_end" not in info:
                    if strict:
                        raise RuntimeError("Efel features require info[stimulus_end]. "
                                           "No stimulus_end found in info, "
                                           "Set stimulus_start, or set strict to "
                                           "False to use end time as stimulus end")
                    else:
                        info["stimulus_end"] = t[-1]
                        self.logger.warning("Efel features require info[stimulus_start]. "
                                            "No stimulus_end found in info, "
                                            "setting stimulus end as end time")

                trace = {}
                trace['T'] = t
                trace['V'] = U
                trace['stim_start'] = [info["stimulus_start"]]
                trace['stim_end'] = [info["stimulus_end"]]


                # Disable decay_time_constant_after_stim if no time points left
                # in simulation after stimulation has ended.
                # Otherwise it thros an error
                if feature_name == "decay_time_constant_after_stim":
                    if info["stimulus_end"] >= t[-1]:
                        return None, None

                result = efel.getMeanFeatureValues([trace], [feature_name], raise_warnings=False)

                return None, result[0][feature_name]

            function.__name__ = feature_name
            return function

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





