import warnings
import numpy as np

def none_to_nan(values):
    """
    Converts ``None`` values in `values` to a arrays of ``np.nan``.

    Parameters
    ----------
    values : array_like, list, number
        Result from model or features. Can be of any dimensions.

    Returns
    -------
    values : array_like, list, number
        `values` with all occurrences of None converted to ``np.nan``.
    """

    if values is None:
        values = np.nan
    else:
        try:
            for i, value in enumerate(values):
                if hasattr(value, "__iter__"):
                    values[i] = none_to_nan(value)

                elif value is None:
                    values[i] = np.nan

        except TypeError:
            return values

    return values



# def none_to_nan_regularize(values):
#     """
#     Converts None values in `values` to a arrays of numpy.nan.

#     If `values` is a 2 dimensional or above array, each instance of None is converted to an
#     array of numpy.nan of the correct shape, which makes the array regular.


#     Parameters
#     ----------
#     values : array_like
#         Result from model or features. Can be of any dimensions.

#     Returns
#     -------
#     array
#         Array with all None converted to arrays of NaN of the correct shape.


#     Examples
#     --------
#     >>> from uncertainpy import Parallel
#     >>> parallel = Parallel()
#     >>> U_irregular = np.array([None, np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])])])
#     >>> result = parallel.none_to_nan(U_irregular)
#         array([[[ nan,  nan,  nan],
#                 [ nan,  nan,  nan],
#                 [ nan,  nan,  nan],
#                 [ nan,  nan,  nan]],
#                 [[ nan,  nan,  nan],
#                 [  1.,   2.,   3.],
#                 [ nan,  nan,  nan],
#                 [  1.,   2.,   3.]]])
#     """
#     warnings.warn(
#         "regularize_nan_results is no longer used as nan results no longer are required to be regular.",
#         DeprecationWarning
#     )

#     is_array = False
#     if isinstance(values, np.ndarray):
#         is_array = True
#         values = values.tolist()

#     if values is None:
#         values = np.nan
#     # elif hasattr(values, "__iter__") and len(values) == 0:
#     #     values_list = np.nan
#     else:
#         # To handle the special case of 0d arrays,
#         # which have an __iter__, but cannot be iterated over
#         try:
#             for i, value in enumerate(values):
#                 if hasattr(value, "__iter__"):
#                     values[i] = none_to_nan_regularize(value)

#             fill = np.nan
#             for i, value in enumerate(values):
#                 if value is not None:
#                     fill = np.full(np.shape(values[i]), np.nan, dtype=float).tolist()
#                     break

#             for i, value in enumerate(values):
#                 if value is None:
#                     values[i] = fill

#         except TypeError:
#             if is_array:
#                 value = np.array(values)

#             return values

#     if is_array:
#         value = np.array(values)

#     return values


def lengths(values):
    """
    Get the lengths of a list and all its sublists.

    Parameters
    ----------
    values : list
        List where we want to find the lengths of the list and all sublists.

    Returns
    -------
    list
        A list with the lengths of the list and all sublists.
    """
    lengths = []

    def recursive_len(values, lengths):
        if hasattr(values, "__iter__"):
            lengths.append(len(values))

            for value in values:
                recursive_len(value, lengths)

    recursive_len(values, lengths)

    return lengths




# def only_contains_none_or_nan(self, values):
#     """
#     Test if all elements in `values` are ``None`` and/or ``numpy.nan``.

#     Parameters
#     ----------
#     values : {list, array_like}
#         A list of values to check if are ``None`` and/or ``numpy.nan``.

#     Returns
#     -------
#     bool
#         True if all elements in `values` are ``None`` and/or ``numpy.nan``.
#     """

#     def recursive_none_nan(values):
#         if hasattr(values, "__iter__"):
#             for value in values:
#                 result = recursive_none_nan(value)

#                 if not result:
#                     return False

#         elif np.isscalar(values):
#             return False

#         else:
#             return True

#     return recursive_none_nan(values)
