from uncertainpy.features import GeneralFeatures
import numpy as np
import elephant.statistics as stat

class NetworkFeatures(GeneralFeatures):
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