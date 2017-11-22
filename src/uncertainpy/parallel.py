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
    Calculate model and features in parallel for one instance of model parameters.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on.
        Default is None.
    features : {None, GeneralFeatures or GeneralFeatures subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if a filename is given.
        Default is None.

    Attributes
    ----------
    model : uncertainpy.Parallel.model
    features : uncertainpy.Parallel.features
    logger : logging.Logger object
        Logger object responsible for logging to screen or file.

    See Also
    --------
    uncertainpy.features.GeneralFeatures : General features class
    uncertainpy.models.Model : Model class
    """
    def create_interpolations(self, result):
        """
        Create an interpolation for adaptive model and features `result`.

        Adaptive model or feature `result`, meaning they
        have a varying number of time steps, are interpolated.
        Interpolation is only performed for one dimensional `result`.
        zero dimensional `result` does not need to be interpolated,
        and support for interpolating two dimensional and above `result`
        have currently not been implemented.

        Parameters
        ----------
        result : dict
            The model and feature results. The model and each feature each has
            a dictionary with the time values, ``"t"``,  and model/feature
            results, ``"U"``.
            An example:

            .. code-block:: Python

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

        Returns
        -------
        result : dict
            If an interpolation has been created, those features/model have
            "interpolation" and the corresponding interpolation object added to
            each features/model dictionary.
            An example:

            .. code-block:: Python

                result = {self.model.name: {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
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
                                             "interpolation": scipy interpolation object},
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
        """
        Run a model and calculate features from the model output,
        return the results.

        The model is run and each feature of the model is calculated from the model output,
        ``t`` (time values) and ``U`` (model result).
        The results are interpolated if they are adaptive, meaning they return a varying number of steps,
        An interpolation is created and added to results for the model/features that are adaptive.
        Each instance of ``None`` is converted to an
        array of ``numpy.nan`` of the correct shape, which makes the array regular.


        Parameters
        ----------
        model_parameters : dictionary
            All model parameters as a dictionary.
            These parameters are sent to model.run().

        Returns
        -------
        result : dictionary
            The model and feature results. The model and each feature each has
            a dictionary with the time values, ``"t"``,  and model/feature results, ``"U"``.
            If an interpolation has been created, those features/model also has
            ``"interpolation"`` added. An example:

            .. code-block:: Python

                result = {self.model.name: {"U": array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
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
                                             "interpolation": scipy interpolation object},
                          "feature_invalid": {"U": np.nan,
                                             "t": np.nan}}


        Notes
        -----
        Time ``t`` and result ``U`` are calculated from the model. Then sent to
        model.postprocess, and the postprocessed result from model.postprocess
        is added to result.
        ``t`` and ``U`` are sent to features.preprocess and the preprocessed results
        is used to calculate each feature.


        See also
        --------
        uncertainpy.Parallel.none_to_nan : Method for converting from None to NaN
        uncertainpy.features.GeneralFeatures.preprocess : preprocessing model results before features are calculated
        uncertainpy.models.Model.postprocess : posteprocessing of model results
        """
        # Try-except to catch exceptions and print stack trace
        try:
            model_result = self.model.run(**model_parameters)

            self.model.validate_run_result(model_result)

            results = {}

            if not self.model.ignore:
                postprocess_result = self.model.postprocess(*model_result)

                try:
                    t_postprocess, U_postprocess = postprocess_result
                except (ValueError, TypeError) as error:
                    msg = "model.postprocess() must return t and U (return t, U | return None, U)"
                    if not error.args:
                        error.args = ("",)
                    error.args = error.args + (msg,)
                    raise

                U_postprocess = self.none_to_nan(U_postprocess)
                t_postprocess = self.none_to_nan(t_postprocess)

                results[self.model.name] = {"t": t_postprocess,
                                            "U": U_postprocess}


            # Calculate features from the model results
            try:
                feature_preprocess = self.features.preprocess(*model_result)
            except TypeError as error:
                msg = "features.preprocess(*model_result) arguments are not the same as the model arguments model.run() returns."
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            try:
                feature_results = self.features.calculate_features(*feature_preprocess)
            except TypeError as error:
                msg = "feature(*model_result) arguments are not the same as the model arguments model.run() returns."
                if not error.args:
                    error.args = ("",)
                error.args = error.args + (msg,)
                raise

            for feature in feature_results:
                t_feature = feature_results[feature]["t"]
                U_feature = feature_results[feature]["U"]

                t_feature = self.none_to_nan(t_feature)
                U_feature = self.none_to_nan(U_feature)

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
        Converts None values in `U` to a arrays of numpy.nan.

        If `U` is a 2 dimensional or above array, each instance of None is converted to an
        array of numpy.nan of the correct shape, which makes the array regular.


        Parameters
        ----------
        U : array_like
            Result from model or features. Can be of any dimensions.

        Returns
        -------
        array
            Array with all None converted to arrays of NaN of the correct shape.


        Examples
        --------
        >>> from uncertainpy import Parallel
        >>> parallel = Parallel()
        >>> U_irregular = np.array([None, np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])])])
        >>> result = parallel.none_to_nan(U_irregular)
            array([[[ nan,  nan,  nan],
                    [ nan,  nan,  nan],
                    [ nan,  nan,  nan],
                    [ nan,  nan,  nan]],
                   [[ nan,  nan,  nan],
                    [  1.,   2.,   3.],
                    [ nan,  nan,  nan],
                    [  1.,   2.,   3.]]])
        """
        U_list = np.array(U).tolist()

        if U is None:
            U_list = np.nan
        elif hasattr(U, "__iter__") and len(U) == 0:
            U_list = np.nan
        else:
            # To handle the special case of 0d arrays,
            # which have an __iter__, but cannot be iterated over
            try:
                for i, u in enumerate(U):
                    if hasattr(u, "__iter__"):
                        U_list[i] = self.none_to_nan(u)

                fill = np.nan
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

