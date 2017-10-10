try:
    import efel

    prerequisites = True
except ImportError:
    prerequisites = False

import types
from .general_features import GeneralFeatures



class EfelFeatures(GeneralFeatures):
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 adaptive=None,
                 labels={}):

        if not prerequisites:
            raise ImportError("Efel features require: efel")

        implemented_labels = {"nr_spikes": ["number of spikes"],
                              "spike_rate": ["spike rate [Hz]"],
                              "time_before_first_spike": ["time [ms]"],
                              "accommodation_index": ["accommodation index"],
                              "average_AP_overshoot": ["voltage [mV]"],
                              "average_AHP_depth": ["voltage [mV]"],
                              "average_AP_width": ["time [ms]"]
                             }

        # new_utility_methods = ["add_efel_features"]

        new_utility_methods = []

        def efel_wrapper(feature_name):
            def function(t, U):
                trace = {}
                trace['T'] = t
                trace['V'] = U
                trace['stim_start'] = [t[0]]
                trace['stim_end'] = [t[-1]]

                result = efel.getFeatureValues([trace], [feature_name])

                return None, result[0][feature_name]

            function.__name__ = feature_name
            return function

        super(EfelFeatures, self).__init__(new_features=new_features,
                                           features_to_run=features_to_run,
                                           adaptive=adaptive,
                                           new_utility_methods=new_utility_methods,
                                           labels=implemented_labels)

        for feature_name in efel.getFeatureNames():
            self.add_features(efel_wrapper(feature_name))

        self.labels = labels
        self.features_to_run = features_to_run





