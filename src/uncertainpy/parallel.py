import traceback

import numpy as np
import scipy.interpolate as scpi

from utils import create_logger
from features import GeneralFeatures
from models import Model

from base import Base

"""
result = {model.name: {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                       "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature1d": {"U": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature0d": {"U": 1,
                        "t": np.nan},
          "feature2d": {"U": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
                        "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
          "feature_adaptive": {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                               "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                               "interpolation": <scipy.interpolate.fitpack2.\
                                                InterpolatedUnivariateSpline\
                                                object at 0x7f1c78f0d4d0>},
          "feature_invalid": {"U": np.nan,
                              "t": np.nan}}
"""


class Parallel(Base):

    def sort_features(self, results):

        """
        result = {"feature1d": {"U": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                  "feature2d": {"U": array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])},
                  self.model.name: {"U": array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
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



    def create_interpolations(self, results):
        features_0d, features_1d, features_2d = self.sort_features(results)

        for feature in features_0d:
            if feature in self.features.adaptive_features or \
                    (feature == self.model.name and self.model.adaptive):
                raise AttributeError("{} is 0D,".format(feature)
                                     + " unable to perform interpolation")

        for feature in features_1d:
            if feature in self.features.adaptive_features or \
                    (feature == self.model.name and self.model.adaptive):
                if np.any(np.isnan(results[feature]["t"])):
                    raise AttributeError("{} does not return any t values.".format(feature)
                                         + " Unable to perform interpolation")

                interpolation = scpi.InterpolatedUnivariateSpline(results[feature]["t"],
                                                                  results[feature]["U"],
                                                                  k=3)
                results[feature]["interpolation"] = interpolation


        for feature in features_2d:
            if feature in self.features.adaptive_features or \
                    (feature == self.model.name and self.model.adaptive):
                raise NotImplementedError("{feature},".format(feature=feature)
                                          + " no support for >= 2D interpolation")

        return results



    def run(self, model_parameters):

        # Try-except to catch exeptions and print stack trace
        try:
            model_result = self.model.run(**model_parameters)

            try:
                # TODO allow for more parameters to be returned, but only the two first are used?
                # use a dictionary for more values?
                # t, U = model_result[:2]
                t, U = model_result
            except ValueError as error:
                msg = "model.run() or model function must return t and U (return t, U | return None, U)"
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            # if U is None:
            #     raise ValueError("U has not been calculated")

            t_postprocess, U_postprocess = self.model.postprocess(t, U)

            if t_postprocess is None:
                t_postprocess = np.nan

            if U_postprocess is None:
                U_postprocess = np.nan

            results = {}
            results[self.model.name] = {"t": t_postprocess, "U": U_postprocess}


            # Calculate features from the model results
            t_preprocess, U_preprocess = self.features.preprocess(t, U)
            feature_results = self.features.calculate_features(t_preprocess, U_preprocess)

            for feature in feature_results:
                t_feature = feature_results[feature]["t"]
                U_feature = feature_results[feature]["U"]

                if t_feature is None:
                    t_feature = np.nan

                if U_feature is None:
                    U_feature = np.nan

                results[feature] = {"U": np.array(U_feature),
                                    "t": np.array(t_feature)}

            # Create interpolations
            results = self.create_interpolations(results)

            return results


        except Exception as e:
            print "Caught exception in parallel run of model:"
            print ""
            traceback.print_exc()
            print ""
            raise e
