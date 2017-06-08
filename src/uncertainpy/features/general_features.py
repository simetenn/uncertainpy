
class GeneralFeatures(object):
    def __init__(self,
                 features_to_run="all",
                 new_utility_methods=None,
                 adaptive_features=None):

        # self.implemented_features = []
        self.utility_methods = ["calculate_feature",
                                "calculate_features",
                                "calculate_all_features",
                                "calculate",
                                "__init__",
                                "implemented_features",
                                "preprocess",
                                "add_features"]

        if new_utility_methods is None:
            new_utility_methods = []

        self.t = None
        self.U = None

        self._features_to_run = None
        self._adaptive_features = None

        self.utility_methods += new_utility_methods

        self.features_to_run = features_to_run
        self.adaptive_features = adaptive_features


    def preprocess(self):
        pass


    @property
    def features_to_run(self):
        return self._features_to_run

    @features_to_run.setter
    def features_to_run(self, new_features_to_run):
        if new_features_to_run == "all":
            self._features_to_run = self.implemented_features()
        elif new_features_to_run is None:
            self._features_to_run = []
        elif isinstance(new_features_to_run, str):
            self._features_to_run = [new_features_to_run]
        else:
            self._features_to_run = new_features_to_run


    @property
    def adaptive_features(self):
        return self._adaptive_features


    @adaptive_features.setter
    def adaptive_features(self, new_adaptive_features):
        if new_adaptive_features == "all":
            self._adaptive_features = self.implemented_features()
        elif new_adaptive_features is None:
            self._adaptive_features = []
        elif isinstance(new_adaptive_features, str):
            self._adaptive_features = [new_adaptive_features]
        else:
            self._adaptive_features = new_adaptive_features


    # TODO is it correct that adding a new feature adds it to features_to_run
    def add_features(self, new_features):
        if callable(new_features):
            setattr(self, new_features.__name__, new_features)
            self._features_to_run.append(new_features.__name__)
        else:
            try:
                for feature in new_features:
                    if callable(feature):
                        setattr(self, feature.__name__, feature)
                        self._features_to_run.append(feature.__name__)
                    else:
                        raise TypeError("feature in iterable is not callable")
            except TypeError as error:
                msg = "features must be a GeneralFeatures instance, callable, list of callables or None"
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise


    def calculate(self, feature_name=None):
        if feature_name is None:
            return self.calculate_features()
        elif feature_name == "all":
            return self.calculate_all_features()
        else:
            feature_result = self.calculate_feature(feature_name)
            try:
                t, U = feature_result
            except ValueError as error:
                msg = "feature_ {} must return t and U (return t, U | return None, U)".format(feature_name)
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            return {feature_name: {"t": t, "U": U}}



    def calculate_feature(self, feature_name):
        if feature_name in self.utility_methods:
            raise TypeError("%s is a utility method")

        return getattr(self, feature_name)(self.t, self.U)



    def calculate_features(self):
        results = {}
        for feature in self.features_to_run:
            feature_result = self.calculate_feature(feature)

            try:
                t, U = feature_result
            except ValueError as error:
                msg = "feature {} must return t and U (return t, U | return None, U)".format(feature)
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            results[feature] = {"t": t, "U": U}

        return results


    def calculate_all_features(self):
        results = {}
        for feature in self.implemented_features():
            feature_result = self.calculate_feature(feature)

            try:
                t, U = feature_result
            except ValueError as error:
                msg = "feature {} must return t and U (return t, U | return None, U)".format(feature)
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            results[feature] = {"t": t, "U": U}


        return results


    def implemented_features(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods and method not in dir(object)]
