from uncertainpy import GeneralFeatures

import numpy as np


class TestingFeatures(GeneralFeatures):
    def __init__(self, features_to_run="all",
                 adaptive_features=["feature_adaptive"]):
        GeneralFeatures.__init__(self,
                                 features_to_run=features_to_run,
                                 adaptive_features=adaptive_features)

        self.is_setup_run = False

    def feature0d(self, t, U):
        return None, 1

    def feature1d(self, t, U):
        return np.arange(0, 10), np.arange(0, 10)

    def feature2d(self, t, U):
        return np.arange(0, 10), np.array([np.arange(0, 10), np.arange(0, 10)])

    def feature_invalid(self, t, U):
        return None, None

    def feature_adaptive(self, t, U):
        return t, U

    def feature_no_time(self, t, U):
        return np.arange(0, 10)

    def setup(self):
        self.is_setup_run = True
