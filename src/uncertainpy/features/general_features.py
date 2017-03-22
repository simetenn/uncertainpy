
class GeneralFeatures():
    def __init__(self, features_to_run="all",
                 new_utility_methods=None,
                 adaptive_features=None):

        # self.implemented_features = []
        self.utility_methods = ["calculateFeature",
                                "calculateFeatures",
                                "calculateAllFeatures",
                                "__init__",
                                "implementedFeatures",
                                "setup"]

        if new_utility_methods is None:
            new_utility_methods = []

        self._t = None
        self._U = None

        self.utility_methods += new_utility_methods


        if features_to_run == "all":
            self.features_to_run = self.implementedFeatures()
        elif features_to_run is None:
            self.features_to_run = []
        elif isinstance(features_to_run, str):
            self.features_to_run = [features_to_run]
        else:
            self.features_to_run = features_to_run


        if adaptive_features == "all":
            self.adaptive_features = self.implementedFeatures()
        elif adaptive_features is None:
            self.adaptive_features = []
        elif isinstance(adaptive_features, str):
            self.adaptive_features = [adaptive_features]
        else:
            self.adaptive_features = adaptive_features



    def setup(self):
        pass


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
            feature_result = self.calculateFeature(feature)
            if (not isinstance(feature_result, tuple) or not isinstance(feature_result, list)) and len(feature_result) != 2:
                raise RuntimeError("feature must return both t and U (return t, U | return None, U)")

            results[feature] = {"t": feature_result[0],
                                "U": feature_result[1]}

        return results


    def calculateAllFeatures(self):
        results = {}
        for feature in self.implementedFeatures():
            feature_result = self.calculateFeature(feature)
            if (not isinstance(feature_result, tuple) or not isinstance(feature_result, list)) and len(feature_result) != 2:
                raise RuntimeError("feature must return both t and U (return t, U | return None, U)")

            results[feature] = {"t": feature_result[0],
                                "U": feature_result[1]}

        return results


    def implementedFeatures(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods]
