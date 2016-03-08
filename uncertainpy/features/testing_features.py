from uncertainpy.features import GeneralFeatures

import numpy as np

class TestingFeatures(GeneralFeatures):
    def feature_0d(self):
        return 1

    def feature_1d(self):
        return np.arange(0, 10)

    def feature_2d(self):
        return np.arrau([np.arange(0, 10), np.arange(0, 10)])
