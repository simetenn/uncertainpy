from uncertainpy import GeneralFeatures

import numpy as np


class TestingFeatures(GeneralFeatures):
    def feature0d(self):
        return None, 1

    def feature1d(self):
        return np.arange(0, 10), np.arange(0, 10)

    def feature2d(self):
        return np.arange(0, 10), np.array([np.arange(0, 10), np.arange(0, 10)])

    def featureInvalid(self):
        return None, None

    def feature_adaptive(self):
        return self.t, self.U
