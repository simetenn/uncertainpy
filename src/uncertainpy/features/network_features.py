from uncertainpy.features import GeneralFeatures
import numpy as np
import elephant.statistics as stat

class NetworkFeatures(GeneralFeatures):
    def __init__(self,
                  features_to_run="all",
                  adaptive_features=None,
                  labels={}):

        implemented_labels = {"cv": ["Neuron", "Coefficient of variation"],
                              "mean cv": ["Coefficient of variation"]
                              }

        super(NetworkFeatures, self).__init__(features_to_run=features_to_run,
                                              adaptive_features=adaptive_features,
                                              labels=implemented_labels)
        self.labels = labels


    def preprocess(self, t, spiketrains):
        return t, spiketrains

    def cv(self, t, spiketrains):
        cv = []
        for spiketrain in spiketrains:
            cv.append(stat.cv(spiketrain))

        return None, np.array(cv)

    def mean_cv(self, t, spiketrains):
        cv = []
        for spiketrain in spiketrains:
            cv.append(stat.cv(spiketrain))

        return None, np.mean(cv)