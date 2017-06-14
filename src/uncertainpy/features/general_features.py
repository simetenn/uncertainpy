
class GeneralFeatures(object):
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 new_utility_methods=None,
                 adaptive=None,
                 labels={}):

        self.utility_methods = ["calculate_feature",
                                "calculate_features",
                                "calculate_all_features",
                                "calculate",
                                "__init__",
                                "implemented_features",
                                "preprocess",
                                "add_features",
                                "serialize"]

        if new_utility_methods is None:
            new_utility_methods = []

        self._features_to_run = None
        self._adaptive = None
        self._labels = {}

        self.utility_methods += new_utility_methods

        self.features_to_run = features_to_run
        self.adaptive = adaptive
        self.labels = labels

        if new_features:
            self.add_features(new_features, labels=labels)




    def preprocess(self, t, U):
        return t, U


    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        self.labels.update(new_labels)


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
    def adaptive(self):
        return self._adaptive


    @adaptive.setter
    def adaptive(self, new_adaptive):
        if new_adaptive == "all":
            self._adaptive = self.implemented_features()
        elif new_adaptive is None:
            self._adaptive = []
        elif isinstance(new_adaptive, str):
            self._adaptive = [new_adaptive]
        else:
            self._adaptive = new_adaptive


    def serialize(feature):
        decorated = True
        for i, spiketrain in enumerate(spiketrains):
            def serialized_feature(t, spiketrains):
                return feature(t, spiketrains[i])

            setattr(self, feature.__name__+ "_i", serialized_feature)





    # TODO is it correct that adding a new feature adds it to features_to_run
    def add_features(self, new_features, labels={}):
        if callable(new_features):
            setattr(self, new_features.__name__, new_features)
            self._features_to_run.append(new_features.__name__)

            tmp_label = labels.get(new_features.__name__)
            if tmp_label is not None:
                self.labels[new_features.__name__] = tmp_label
        else:
            try:
                for feature in new_features:
                    if callable(feature):
                        setattr(self, feature.__name__, feature)
                        self._features_to_run.append(feature.__name__)

                        tmp_lables = labels.get(feature.__name__)
                        if tmp_lables is not None:
                            self.labels[feature.__name__] = tmp_lables
                    else:
                        raise TypeError("Feature in iterable is not callable")
            except TypeError as error:
                msg = "Added features must be a GeneralFeatures instance, callable or list of callables"
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise




    def calculate(self, t, U, feature_name=None):
        if feature_name is None:
            return self.calculate_features(t, U)
        elif feature_name == "all":
            return self.calculate_all_features(t, U)
        else:
            feature_result = self.calculate_feature(t, U, feature_name)
            try:
                feature_t, feature_U = feature_result
            except ValueError as error:
                msg = "feature_ {} must return t and U (return t, U | return None, U)".format(feature_name)
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            return {feature_name: {"t": feature_t, "U": feature_U}}



    def calculate_feature(self, t, U, feature_name):
        if feature_name in self.utility_methods:
            raise TypeError("%s is a utility method")

        return getattr(self, feature_name)(t, U)



    def calculate_features(self, t, U):
        results = {}
        for feature in self.features_to_run:
            feature_result = self.calculate_feature(t, U, feature)

            try:
                feature_t, feature_U = feature_result
            except ValueError as error:
                msg = "feature {} must return t and U (return t, U | return None, U)".format(feature)
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            results[feature] = {"t": feature_t, "U": feature_U}

        return results


    def calculate_all_features(self, t, U):
        results = {}
        for feature in self.implemented_features():
            feature_result = self.calculate_feature(t, U, feature)

            try:
                feature_t, feature_U = feature_result
            except ValueError as error:
                msg = "feature {} must return t and U (return t, U | return None, U)".format(feature)
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            results[feature] = {"t": feature_t, "U": feature_U}


        return results


    def implemented_features(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods and method not in dir(object)]
