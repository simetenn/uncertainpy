from uncertainpy.features.spikes import Spikes
from uncertainpy.features import GeneralFeatures


class GeneralNeuronFeatures(GeneralFeatures):
    def __init__(self, features_to_run="all", adaptive_features=None, thresh=-30, extended_spikes=False):
        new_utility_methods = ["calculateSpikes"]

        GeneralFeatures.__init__(self,
                                 features_to_run=features_to_run,
                                 adaptive_features=adaptive_features,
                                 new_utility_methods=new_utility_methods)

        self.spikes = None

        self.thresh = thresh
        self.extended_spikes = extended_spikes



    def setup(self):
        print "setup"
        self.calculateSpikes(thresh=self.thresh, extended_spikes=self.extended_spikes)



    def calculateSpikes(self, thresh=-30, extended_spikes=False):
        if self.t is None:
            raise AttributeError("t is not assigned")
        if self.U is None:
            raise AttributeError("U is not assigned")

        self.spikes = Spikes()
        self.spikes.detectSpikes(self.t, self.U, thresh=thresh, extended_spikes=extended_spikes)
