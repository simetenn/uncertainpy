import traceback

import numpy as np
import scipy.interpolate as scpi

from .base import Base

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

# TODO test what happens with inherited docstring
class Parallel(Base):
    """
    Evaluates model and features in parallel.


    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}
        Model to perform uncertainty quantification on.
    features : {None, GeneralFeatures or GeneralFeatures subclass instance, list of feature functions}
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if set. Default is None.

    Attributes
    ----------
    model
    features
    logger : logging.Logger object
        Logger object responsible for logging to screen or file.
    """
    def create_interpolations(self, result):
        """
        Create an interpolation for adaptive model and features `result`.

        Adaptive model or feature `result`, meaning they
        have a varying number of time steps, are interpolated.
        Interpolation is only performed for one dimensional `result`s.
        zero dimensional `result`s does not need to be interpolated,
        and support for interpolating two dimensional and above `result`s
        have currently not been implemented.

        Parameters
        ----------
        result : dict
            The model and feature results. The model and each feature each has
            a dictionary with the time values, "t",  and model/feature results, "U".
            An example:

            .. code-block::
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
                                            "t": array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
                        "feature_invalid": {"U": np.nan,
                                            "t": np.nan}}
        Notes
        -----
        If either model or feature results are adaptive,
        the results  must be interpolated for Chaospy
        to be able to create the polynomial approximation.
        For 1D results this is done with scipy:
        ``InterpolatedUnivariateSpline(time, U, k=3)``.
        """

        for feature in result:
            if np.ndim(result[feature]["U"]) == 0:
                if feature in self.features.adaptive or \
                        (feature == self.model.name and self.model.adaptive):
                    raise AttributeError("{} is 0D,".format(feature)
                                         + " interpolation makes no sense.")

            if np.ndim(result[feature]["U"]) == 1:
                if feature in self.features.adaptive or \
                    (feature == self.model.name and self.model.adaptive):
                    if np.any(np.isnan(result[feature]["t"])):
                        raise AttributeError("{} does not return any t values.".format(feature)
                                             + " Unable to perform interpolation.")

                    interpolation = scpi.InterpolatedUnivariateSpline(result[feature]["t"],
                                                                      result[feature]["U"],
                                                                      k=3)
                    result[feature]["interpolation"] = interpolation


            if np.ndim(result[feature]["U"]) >= 2:
                # TODO implement interpolation of >= 2d data, part 1
                if feature in self.features.adaptive or \
                        (feature == self.model.name and self.model.adaptive):
                    raise NotImplementedError("{feature},".format(feature=feature)
                                              + " no support for >= 2D interpolation")

        return result



    def run(self, model_parameters):

        # Try-except to catch exceptions and print stack trace
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

            U_postprocess = self.none_to_nan(U_postprocess)
            t_postprocess = self.none_to_nan(t_postprocess)

            results = {}
            results[self.model.name] = {"t": np.array(t_postprocess),
                                        "U": np.array(U_postprocess)}


            # Calculate features from the model results
            t_preprocess, U_preprocess = self.features.preprocess(t, U)

            feature_results = self.features.calculate_features(t_preprocess, U_preprocess)

            for feature in feature_results:
                t_feature = feature_results[feature]["t"]
                U_feature = feature_results[feature]["U"]

                t_feature = self.none_to_nan(t_feature)|
                U_feature = self.none_to_nan(U_feature)|

                results[feature] = {"U": U_feature,
                                    "t": t_feature}

            # Create interpolations
            results = self.create_interpolations(results)

            return results


        except Exception as e:
            print("Caught exception in parallel run of model:")
            print("")
            traceback.print_exc()
            print("")
            raise e



    def none_to_nan(self, U):
        """
        Converts None values in `U` to a arrays of np.nan.

        Parameters
        ----------
        U : array_like
            Result from model or features. Can be of any dimensions.

        Returns
        -------
        array
            Array with all None converted to arrays of NaN of the correct shape.


        """
        U_list = np.array(U).tolist()

        if U is None:
            U_list = [np.nan]
        else:
            # To handle the special case of 0d arrays, which have an __iter__, but cannot be iterated over
            try:
                for i, u in enumerate(U):
                    if hasattr(u, "__iter__"):
                        U_list[i] = self.none_to_nan(u)

                for i, u in enumerate(U):
                    if u is not None:
                        fill = np.full(np.shape(U_list[i]), np.nan, dtype=float).tolist()
                        break

                for i, u in enumerate(U):
                    if u is None:
                        U_list[i] = fill

            except TypeError:
                return U_list


        return np.array(U_list)

