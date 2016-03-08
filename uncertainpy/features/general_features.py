import os
import sys
import numpy as np


class GeneralFeatures():
    def __init__(self, t=None, U=None, new_utility_methods=None):
        self.t = t
        self.U = U

        # self.implemented_features = []
        self.utility_methods = ["calculateFeature",
                                "calculateFeatures",
                                "calculateAllFeatures",
                                "__init__",
                                "implementedFeatures",
                                "cmd"]

        if new_utility_methods is None:
            new_utility_methods = []

        self.utility_methods = self.utility_methods + new_utility_methods

        self.filepath = sys.modules[self.__class__.__module__].__file__
        self.filedir = os.path.dirname(self.filepath)
        self.filename = os.path.basename(self.filepath)

        if self.__class__.__module__ == "__main__":
            self.filedir = os.path.dirname(os.path.abspath(self.filename))


    def cmd(self):
        return self.filedir, self.filename, self.__class__.__name__


    def calculateFeature(self, feature_name):
        if not callable(getattr(self, feature_name)):
            raise NotImplementedError("%s is not a implemented feature" % (feature_name))

        tmp_result = getattr(self, feature_name)()
        if tmp_result is None:
            return np.NaN
        else:
            return tmp_result

    def calculateFeatures(self, feature_names):
        results = {}
        for feature in feature_names:
            results[feature] = self.calculateFeature(feature)

        return results


    def calculateAllFeatures(self):
        results = {}
        for feature in self.implementedFeatures():
            results[feature] = self.calculateFeature(feature)

        return results


    def implementedFeatures(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods]
