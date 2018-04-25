from uncertainpy import Features

import numpy as np


class TestingFeatures(Features):
    def __init__(self, features_to_run="all",
                 interpolate=["feature_interpolate"]):
        Features.__init__(self,
                                 features_to_run=features_to_run,
                                 interpolate=interpolate)

        implemented_labels = {"feature0d": ["feature0d"],
                              "feature1d": ["feature1d x", "feature1d y"],
                              "feature2d": ["feature2d x", "feature2d y", "feature2d z"],
                              "feature0d_var": ["feature0d"],
                              "feature1d_var": ["feature1d x", "feature1d y"],
                              "feature2d_var": ["feature2d x", "feature2d y", "feature2d z"]
                             }

        super(TestingFeatures, self).__init__(features_to_run=features_to_run,
                                              interpolate=interpolate,
                                              labels=implemented_labels)

        self.is_preprocess_run = False

    def feature0d(self, time, values):
        return None, 1

    def feature1d(self, time, values):
        return np.arange(0, 10), np.arange(0, 10)

    def feature2d(self, time, values):
        return np.arange(0, 10), np.array([np.arange(0, 10), np.arange(0, 10)])

    def feature0d_var(self, time, values):
        return None, 1 + np.mean(values)

    def feature1d_var(self, time, values):
        return np.arange(0, 10), np.arange(0, 10) + + np.mean(values)

    def feature2d_var(self, time, values):
        return np.arange(0, 10), np.array([np.arange(0, 10), np.arange(0, 10)]) + + np.mean(values)


    def feature_invalid(self, time, values):
        return None, None

    def feature_interpolate(self, time, values):
        return time, values

    def feature_no_time(self, time, values):
        return np.arange(0, 10)

    def preprocess(self, time, values):
        self.is_preprocess_run = True
        return time, values

    def feature_info(self, time, values, info):
        self.info = info
        return time, values

    def feature_error_one(self):
        return 1

    def feature_error_value(self):
        return (1, 2, 3)
