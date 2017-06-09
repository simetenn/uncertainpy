from uncertainpy.features.spikes import Spikes
from uncertainpy.features import GeneralFeatures


class GeneralSpikingFeatures(GeneralFeatures):
    def __init__(self,
                 features_to_run="all",
                 adaptive_features=None,
                 thresh=-30,
                 extended_spikes=False,
                 labels={}):

        new_utility_methods = ["calculate_spikes"]

        super(GeneralSpikingFeatures, self).__init__(features_to_run=features_to_run,
                                                     adaptive_features=adaptive_features,
                                                     new_utility_methods=new_utility_methods,
                                                     labels=labels)

        self.spikes = None

        self.thresh = thresh
        self.extended_spikes = extended_spikes



    def preprocess(self, t, U):
        self.U = U

        self.calculate_spikes(t, U, thresh=self.thresh, extended_spikes=self.extended_spikes)

        return t, self.spikes


    def calculate_spikes(self, t, U, thresh=-30, extended_spikes=False):
        self.spikes = Spikes()
        self.spikes.find_spikes(t, U, thresh=thresh, extended_spikes=extended_spikes)
