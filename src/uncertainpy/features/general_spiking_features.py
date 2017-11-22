from .spikes import Spikes
from .general_features import GeneralFeatures


class GeneralSpikingFeatures(GeneralFeatures):
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 adaptive=None,
                 thresh=-30,
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

        self.thresh = thresh
        self.extended_spikes = extended_spikes



    def preprocess(self, t, U, info):
        self.U = U

        self.calculate_spikes(t, U, thresh=self.thresh, extended_spikes=self.extended_spikes)

        return t, self.spikes, info


    def calculate_spikes(self, t, U, thresh=-30, extended_spikes=False):
        self.spikes = Spikes()
        self.spikes.find_spikes(t, U, thresh=thresh, extended_spikes=extended_spikes)
