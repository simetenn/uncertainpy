class GeneralFeatures(object):
    """
    Class for calculating features of a model.

    Parameters
    ----------
    new_features : {None}, optional
        A list if features
    features_to_run : {"all", None, str, list of feature names}, optional
        Which features to calculate uncertainties for.
        If ``"all"``, the uncertainties will be calculated for all
        implemented and assigned features.
        If ``None``, or an empty list ``[]``, no features will be
        calculated.
        If str, only that feature ic calculated.
        If list of feature names, all the listed features will be
        calculated. Default is ``"all"``.
    new_utility_methods : {None}, optional
    adaptive : {None}, optional
    labels : dictionary, optional

    Attributes
    ----------


    """
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
                                "add_features"]

        if new_utility_methods is None:
            new_utility_methods = []

        self._features_to_run = []
        self._adaptive = None
        self._labels = {}

        self.utility_methods += new_utility_methods


        self.adaptive = adaptive

        if new_features is not None:
            self.add_features(new_features, labels=labels)

        self.labels = labels
        self.features_to_run = features_to_run



    def preprocess(self, *args):
        """
        """
        return args


    @property
    def labels(self):
        """
        Labels for the axes of each feature, used when plotting.

        Parameters
        ----------
        new_labels : dictionary
            A dictionary with key as the feature name and the value a list of
            labels for each axis `
            {"nr_spikes": ["number of spikes"],
                              "spike_rate": ["spike rate [Hz]"],
                              "time_before_first_spike": ["time [ms]"],
                              "accommodation_index": ["accommodation index"],
                              "average_AP_overshoot": ["voltage [mV]"],
                              "average_AHP_depth": ["voltage [mV]"],
                              "average_AP_width": ["time [ms]"]
                             }
        """
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        self.labels.update(new_labels)


    @property
    def features_to_run(self):
        """
        Which features to calculate uncertainties for.

        Parameters
        ----------
        new_features_to_run : {"all", None, str, list of feature names}
            Which features to calculate uncertainties for.
            If ``"all"``, the uncertainties will be calculated for all
            implemented and assigned features.
            If ``None``, or an empty list , no features will be
            calculated.
            If ``str``, only that feature is calculated.
            If ``list`` of feature names, all listed features will be
            calculated. Default is ``"all"``.

        Returns
        -------
        list
            A list of features to calculate uncertainties for.

        """

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
        """
        """
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



    def add_features(self, new_features, labels={}):
        if callable(new_features):
            setattr(self, new_features.__name__, new_features)
            # self.features_to_run.append(new_features.__name__)

            tmp_label = labels.get(new_features.__name__)
            if tmp_label is not None:
                self.labels[new_features.__name__] = tmp_label
        else:
            try:
                for feature in new_features:
                    if callable(feature):
                        setattr(self, feature.__name__, feature)
                        # self.features_to_run.append(feature.__name__)

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




    # def calculate(self, feature_name=None, *args):
    #     if feature_name is None:
    #         return self.calculate_features(*args)
    #     elif feature_name == "all":
    #         return self.calculate_all_features(*args)
    #     else:
    #         feature_result = self.calculate_feature(feature_name, *args)
    #         try:
    #             feature_t, feature_U = feature_result
    #         except ValueError as error:
    #             msg = "feature_ {} must return t and U (return t, U | return None, U)".format(feature_name)
    #             if not error.args:
    #                 error.args = ("",)
    #             error.args = error.args + (msg,)
    #             raise

    #         return {feature_name: {"t": feature_t, "U": feature_U}}



    def calculate_feature(self, feature_name, *args):
        if feature_name in self.utility_methods:
            raise TypeError("%s is a utility method")

        feature_result = getattr(self, feature_name)(*args)

        try:
            feature_t, feature_U = feature_result
        except (ValueError, TypeError) as error:
            msg = "feature {} must return t and U (return t, U | return None, U)".format(feature_name)
            if not error.args:
                error.args = ("",)
            error.args = error.args + (msg,)
            raise

        return feature_result #getattr(self, feature_name)(*args)



    def calculate_features(self, *args):
        results = {}
        for feature in self.features_to_run:
            feature_t, feature_U = self.calculate_feature(feature, *args)

            results[feature] = {"t": feature_t, "U": feature_U}

        return results


    def calculate_all_features(self, *args):
        results = {}
        for feature in self.implemented_features():
            feature_t, feature_U = self.calculate_feature(feature, *args)

            results[feature] = {"t": feature_t, "U": feature_U}

        return results


    def implemented_features(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods and method not in dir(object)]
