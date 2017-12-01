import os
import h5py
import collections

import numpy as np

from .utils import create_logger


class DataFeature(collections.MutableMapping):
    def __init__(self,
                 name,
                 U=None,
                 time=None,
                 E=None,
                 Var=None,
                 p_05=None,
                 p_95=None,
                 sensitivity_1=None,
                 total_sensitivity_1=None,
                 sensitivity_t=None,
                 total_sensitivity_t=None,
                 labels=None):

        self.name = name
        self.values = U
        self.time = time
        self.E = E
        self.Var = Var
        self.p_05 = p_05
        self.p_95 = p_95
        self.sensitivity_1 = sensitivity_1
        self.total_sensitivity_1 = total_sensitivity_1
        self.sensitivity_t = sensitivity_t
        self.total_sensitivity_t = total_sensitivity_t
        self.labels = labels

        self._built_in_data_types = ["values", "time", "E", "Var", "p_05", "p_95",
                                    "sensitivity_1", "total_sensitivity_1",
                                    "sensitivity_t", "total_sensitivity_t", "labels"]

        self._information = ["name", "labels"]

    def __getitem__(self, data_type):
        # if data_type not in self.built_in_data_types:
        #     raise AttributeError("{} is not a supported attribute".format(data_type))

        return getattr(self, data_type)


    def get_data_types(self):
        data_types = []

        for data_type in dir(self):
            if not data_type.startswith('_') and not callable(self[data_type]) \
                and self[data_type] is not None and data_type not in self._information:
                data_types.append(data_type)

        return data_types


    def __setitem__(self, data_type, value):
        # if data_type not in self.built_in_data_types:
        #     raise AttributeError("{} is not a supported attribute".format(data_type))

        setattr(self, data_type, value)


    def __iter__(self):
        for data_type in self.get_data_types():
            yield data_type

        # for data_type in self.get_data_types():
        #     if self[data_type] is not None:
        #         yield self[data_type]


    def __delitem__(self, data_type):
        # if data_type not in self.built_in_data_types:
        #     raise AttributeError("{} is not a supported attribute".format(data_type))

        setattr(self, data_type, None)


    def __len__(self):
        return len(self.get_data_types())


    def __contains__(self, data_type):
        if data_type not in self.get_data_types() or self[data_type] is None:
            return False
        else:
            return True


    def ndim(self):
        """
        Get the number of dimensions of a `feature`.

        Parameters
        ----------
        feature : str
            Name of the model or a feature.

        Returns
        -------
        int
            The number of dimensions of the model/feature result.
        """

        if self.values is not None:
            return np.ndim(self.U[0])
        else:
            return None


# TODO instead of a data object, could just a  h5py file have been used ?
class Data(collections.MutableMapping):
    """
    Store data calculated from the uncertainty quantification.

    Has all standard dictionary methods, such as items, value contains
    and so implemented.
    Can be indexed as a regular dictionary with model/feature names
    as keys and returns a dictionary with all data types as values.

    Parameters
    ----------
    filename : str, optional
        Name of the file to load data from. If None, no data is loaded.
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
    model_name : str
        Name of the model.
    data : dictionary
        A dictionary with all data stored.
    logger : logging.Logger object
        Logger object responsible for logging to screen or file.

    Notes
    -----

    Each feature and the model has the following data:

    values : array_like
        Feature or model result.
    time : array_like
        Time values for feature or model.
    E : array_like
        Mean of the feature or model results.
    Var : array_like
        Variance of the feature or model results.
    p_05 : array_like
        5 percentile of the feature or model results.
    p_95 : array_like
        95 percentile of the feature or model results.
    sensitivity_1 : array_like
        First order sensitivity of the feature or model results.
    total_sensitivity_1 : array_like
        First order sensitivity of the feature or model results.
    sensitivity_t : array_like
        Total effect sensitivity of the feature or model results.
    total_sensitivity_t : array_like
        Normalized sum of total effect sensitivity of
        the feature or model results.
    labels : list
        A list of labels for plotting, ``[x-axis, y-axis, z-axis]``

    These are the keys that are found in the
    dictionaries for each model/feature.
    """
    def __init__(self,
                 filename=None,
                 verbose_level="info",
                 verbose_filename=None):

        # self.data_types = ["values", "time", "E", "Var", "p_05", "p_95",
        #                    "sensitivity_1", "total_sensitivity_1",
        #                    "sensitivity_t", "total_sensitivity_t", "labels"]


        self.data_information = ["uncertain_parameters", "model_name",
                                 "incomplete"]

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


        self.uncertain_parameters = []
        self.model_name = ""
        self.incomplete = []
        self.data = {}


        if filename is not None:
            self.load(filename)



    def __str__(self):
        """
        Convert all data to a readable string.

        Returns
        -------
        str
           A readable string of all parameter objects.
        """

        def border(msg):
            count = len(msg) + 6
            line = "="*(count + 2)
            string = """
{line}
|   {msg}   |
{line}\n\n""".format(line=line, msg=msg)
            return string

        output_str = border("Information")

        for info in self.data_information:
            current_info = getattr(self, info)
            output_str += "{info}: {current_info}\n".format(info=info,
                                                            current_info=current_info)

        for feature in self:
            output_str += border(feature)
            output_str += "=== labels ===\n"
            output_str += "{data}\n\n".format(data=self[feature].labels)

            for data_type in self[feature]:
                output_str += "=== {data_type} ===\n".format(data_type=data_type)
                output_str += "{data}\n\n".format(data=self[feature][data_type])


        return output_str.strip()



    def clear(self):
        """
        Clear all data.
        """
        self.uncertain_parameters = []
        self.model_name = ""

        self.data = {}


    def ndim(self, feature):
        """
        Get the number of dimensions of a `feature`.

        Parameters
        ----------
        feature : str
            Name of the model or a feature.

        Returns
        -------
        int
            The number of dimensions of the model/feature result.
        """

        return self[feature].ndim()



    def get_labels(self, feature):
        """
        Get labels for a `feature`. If no labels are defined,
        returns a list with the correct number of empty strings.

        Parameters
        ----------
        feature : str
            Name of the model or a feature.

        Returns
        -------
        list
            A list of labels for plotting, ``[x-axis, y-axis, z-axis]``.
            If no labels are defined,
            returns a list with the correct number of empty strings.
        """
        if self[feature].labels is not None:
            return self[feature].labels

        elif self[self.model_name].labels is not None and self[self.model_name].ndim() == self[feature].ndim():
            return self[self.model_name].labels

        else:
            return [""]*(self[feature].ndim() + 1)



    def __getitem__(self, feature):
        """
        Get dictionary with data for `feature`.

        Parameters
        ----------
        feature: str
            Name of feature/model.

        Returns
        -------
        dictionary
            Dictionary with data for `feature`.
        """
        return self.data[feature]

    def __setitem__(self, feature, data):
        """
        Set dictionary with `data` for `feature`.

        Parameters
        ----------
        feature: str
            Name of feature/model.
        data : dictionary
            Dictionary with data for `feature`.

        Raises
        ------
        ValueError
            If `data` is not a dictionary.
        """
        if not isinstance(data, DataFeature):
            raise ValueError("data must be of type DataFeature")
        self.data[feature] = data


    def __iter__(self):
        """
        Iterate over each feature/model.

        Yields
        ------
        str
            Name of feature/model.
        """
        return iter(self.data)


    def __delitem__(self, feature):
        """
        Delete data for `feature`.

        Parameters
        ----------
        feature: str
            Name of feature.
        """

        del self.data[feature]


    def __len__(self):
        """
        Get the number of model/features.

        Returns
        ----------
        int
            The number of model/features.
        """
        return len(self.data)


    def add_features(self, features):
        """
        Add features to data.abs

        Parameters
        ----------
        features : {str, list}
            Name of feature to add, or list of features to add.
        """
        if isinstance(features, str):
            features = [features]

        for feature in features:
            self.data[feature] = DataFeature(feature)


    def save(self, filename):
        """
        Save data to a hdf5 file with name `filename`.

        Parameters
        ----------
        filename : str
            Name of the file to load data from.
        """
        ### TODO expand the save function to also save parameters and model information

        with h5py.File(filename, 'w') as f:
            f.attrs["uncertain parameters"] = self.uncertain_parameters
            f.attrs["model name"] = self.model_name
            f.attrs["incomplete results"] = self.incomplete

            for feature in self:
                group = f.create_group(feature)

                for data_type in self[feature]:
                    group.create_dataset(data_type, data=self[feature][data_type])

                group.create_dataset("labels", data=self[feature].labels)



    def load(self, filename):
        """
        Load data from a hdf5 file with name `filename`.

        Parameters
        ----------
        filename : str
            Name of the file to load data from.
        """

        # TODO add this check when changing to python 3
        # if not os.path.isfile(self.filename):
        #     raise FileNotFoundError("{} file not found".format(self.filename))
        self.clear()

        with h5py.File(filename, 'r') as f:
            self.uncertain_parameters = list(f.attrs["uncertain parameters"])
            self.model_name = f.attrs["model name"]
            self.incomplete = f.attrs["incomplete results"]

            for feature in f:
                self.add_features(str(feature))
                for data_type in f[feature]:
                    self[feature][data_type] = f[feature][data_type][()]


    def remove_only_invalid_features(self):
        """
        Remove all features that only have invalid results (NaN).
        """
        feature_list = self.data.keys()[:]
        for feature in feature_list:
            all_nan = True
            for U in self[feature]["values"]:
                if not np.all(np.isnan(U)):
                    all_nan = False

            if all_nan:
                self.logger.warning("Feature: {} does".format(feature)
                                    + " not yield results for any parameter combinations")

                del self[feature]



    # # TODO rewrite so this is using np.nan. Currently this makes it so when loading a data file it is loaded as an object
    # def nan_to_none(self, array):
    #     try:
    #         tmp_array = array.astype(object)
    #         tmp_array[np.isnan(array)] = None

    #         return tmp_array

    #     except TypeError:
    #         return array


    # def none_to_nan(self, array):
    #     tmp_array = np.array(array, dtype=float)

    #     return tmp_array


    # def all_to_none(self):
    #     for data_name in self.data_types:
    #         data = getattr(self, data_name)
    #         if data is not None:
    #             for feature in data:
    #                 print data[feature]
    #                 data[feature] = self.nan_to_none(data[feature])
    #
    #         data = self.nan_to_none(data)
    #
    # def all_to_nan(self):
    #     for data_name in self.data_types:
    #         data = getattr(self, data_name)
    #         if data is not None:
    #             for feature in data:
    #                 data[feature] = self.none_to_nan(data[feature])
    #
    #         data = self.nan_to_none(data)


