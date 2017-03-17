import os
import sys


class GeneralFeatures():
    def __init__(self, features_to_run="all",
                 new_utility_methods=None,
                 adaptive=None):

        # self.implemented_features = []
        self.utility_methods = ["calculateFeature",
                                "calculateFeatures",
                                "calculateAllFeatures",
                                "__init__",
                                "implementedFeatures"]#,
                                # "cmd",
                                # "kwargs",
                                # "set_properties"]

        if new_utility_methods is None:
            new_utility_methods = []

        self._t = None
        self._U = None

        self.utility_methods += new_utility_methods

        # self.filepath = sys.modules[self.__class__.__module__].__file__
        # self.filedir = os.path.dirname(self.filepath)
        # self.filename = os.path.basename(self.filepath)
        #
        # if self.__class__.__module__ == "__main__":
        #     self.filedir = os.path.dirname(os.path.abspath(self.filename))


        if features_to_run == "all":
            self.features_to_run = self.implementedFeatures()
        elif features_to_run is None:
            self.features_to_run = []
        elif isinstance(features_to_run, str):
            self.features_to_run = [features_to_run]
        else:
            self.features_to_run = features_to_run

        self.additional_kwargs = []


    # def set_properties(self, kwargs):
    #     self.additional_kwargs = kwargs.keys()
    #     for cmd in self.additional_kwargs:
    #         if hasattr(self, cmd):
    #             raise RuntimeWarning("{} already have attribute {}".format(self.__class__.__name__, cmd))
    #
    #         setattr(self, cmd, kwargs[cmd])


    # def cmd(self):
    #     return {"filedir": self.filedir,
    #             "filename": self.filename,
    #             "feature_name": self.__class__.__name__}
    #
    #
    #
    # def kwargs(self):
    #     kwargs = {"features_to_run": self.features_to_run}
    #
    #     for kwarg in self.additional_kwargs:
    #         kwargs[kwarg] = getattr(self, kwarg)
    #
    #     return kwargs

    @property
    def t(self):
        return self._t

    @property
    def U(self):
        return self._U

    def calculateFeature(self, feature_name):
        if feature_name in self.utility_methods:
            raise TypeError("%s is a utility method")

        # if not callable(getattr(self, feature_name)):
        #     raise NotImplementedError("%s is not a implemented feature" % (feature_name))

        return getattr(self, feature_name)()



    def calculateFeatures(self):
        results = {}
        for feature in self.features_to_run:
            t, U = self.calculateFeature(feature)
            results[feature] = {"t": t, "U": U}

        return results


    def calculateAllFeatures(self):
        results = {}
        for feature in self.implementedFeatures():
            t, U = self.calculateFeature(feature)
            results[feature] = {"t": t, "U": U}

        return results


    def implementedFeatures(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods]
