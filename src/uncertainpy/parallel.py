import traceback

import numpy as np
import scipy.interpolate as scpi

from utils import create_logger
from features import GeneralFeatures
from models import Model


"""
result = {"directComparison": {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                               "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature1d": {"U": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature0d": {"U": 1,
                        "t": None},
          "feature2d": {"U": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature_adaptive": {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                               "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                               "interpolation": <scipy.interpolate.fitpack2.\
                                                InterpolatedUnivariateSpline\
                                                object at 0x7f1c78f0d4d0>},
          "featureInvalid": {"U": None,
                             "t": None}}
"""


class Parallel(object):
    def __init__(self,
                 model,
                 features=None,
                 base_features=GeneralFeatures,
                 verbose_level="info",
                 verbose_filename=None):

        self._features = None
        self._model = None


        self.base_features = base_features
        self.features = features
        self.model = model


        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


    @property
    def features(self):
        return self._features


    @features.setter
    def features(self, new_features):
        if new_features is None:
            self._features = self.base_features(features_to_run=None)
        elif isinstance(new_features, GeneralFeatures):
            self._features = new_features
        else:
            self._features = self.base_features(features_to_run="all")
            self._features.add_features(new_features)
            self._features.features_to_run = "all"


    @property
    def model(self):
        return self._model


    @model.setter
    def model(self, new_model):
        if isinstance(new_model, Model) or new_model is None:
            self._model = new_model
        elif callable(new_model):
            self._model = Model()
            self._model.run = new_model
        else:
            raise TypeError("model must be a Model instance, callable or None")



    def sortFeatures(self, results):

        """
        result = {"feature1d": {"U": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                  "feature2d": {"U": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])},
                  "directComparison": {"U": array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
                                       "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                  "feature0d": {"U": 1}}
        """

        features_2d = []
        features_1d = []
        features_0d = []

        for feature in results:
            if hasattr(results[feature]["U"], "__iter__"):
                if len(results[feature]["U"].shape) == 0:
                    features_0d.append(feature)
                elif len(results[feature]["U"].shape) == 1:
                    features_1d.append(feature)
                else:
                    features_2d.append(feature)
            else:
                features_0d.append(feature)

        return features_0d, features_1d, features_2d



    def createInterpolations(self, results):
        features_0d, features_1d, features_2d = self.sortFeatures(results)

        for feature in features_0d:
            if feature in self.features.adaptive_features or \
                    (feature == "directComparison" and self.model.adaptive_model):
                raise AttributeError("{} is 0D,".format(feature)
                                     + " unable to perform interpolation")

        for feature in features_1d:
            if feature in self.features.adaptive_features or \
                    (feature == "directComparison" and self.model.adaptive_model):
                if results[feature]["t"] is None:
                    raise AttributeError("{} does not return any t values.".format(feature)
                                         + " Unable to perform interpolation")

                interpolation = scpi.InterpolatedUnivariateSpline(results[feature]["t"],
                                                                  results[feature]["U"],
                                                                  k=3)
                results[feature]["interpolation"] = interpolation


        for feature in features_2d:
            if feature in self.features.adaptive_features or \
                    (feature == "directComparison" and self.model.adaptive_model):
                raise NotImplementedError("{feature},".format(feature=feature)
                                          + " no support for >= 2D interpolation")

        return results



    def run(self, model_parameters):

        # Try-except to catch exeptions and print stack trace
        try:
            model_result = self.model.run(**model_parameters)

            try:
                t, U = model_result
            except ValueError as error:
                msg = "model.run() must return t and U (return t, U | return None, U)"
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            if U is None:
                raise ValueError("U has not been calculated")

            # if np.all(np.isnan(t)):
            #     t = None


            results = {}
            results["directComparison"] = {"t": t, "U": U}

            # Calculate features from the model results
            self.features.t = t
            self.features.U = U
            self.features.setup()
            feature_results = self.features.calculateFeatures()


            for feature in feature_results:
                feature_t = feature_results[feature]["t"]
                if feature_t is not None and np.all(np.isnan(feature_t)):
                    feature_t = None

                results[feature] = {"U": feature_results[feature]["U"],
                                    "t": feature_t}

            # Create interpolations
            results = self.createInterpolations(results)

            return results


        except Exception as e:
            print("Caught exception in parallel run of model:")
            print("")
            traceback.print_exc()
            print("")
            raise e
