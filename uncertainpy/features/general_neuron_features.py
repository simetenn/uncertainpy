from uncertainpy.spikes import Spikes
from uncertainpy.features import GeneralFeatures


class GeneralNeuronFeatures(GeneralFeatures):
    def __init__(self, t=None, U=None, thresh=-30, extended_spikes=False):
        new_utility_methods = ["calculateSpikes"]

        GeneralFeatures.__init__(self, t=t, U=U, new_utility_methods=new_utility_methods)

        self.spikes = None

        if self.t is not None and self.U is not None:
            self.calculateSpikes(thresh=thresh, extended_spikes=extended_spikes)


    def calculateSpikes(self, thresh=-30, extended_spikes=False):
        if self.t is None:
            raise AttributeError("t is not assigned")
        if self.U is None:
            raise AttributeError("V is not assigned")

        self.spikes = Spikes()
        self.spikes.detectSpikes(self.t, self.U, thresh=thresh, extended_spikes=extended_spikes)
