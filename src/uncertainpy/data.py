import os
import h5py
import collections

import numpy as np

from .utils import create_logger


class DataFeature(collections.MutableMapping):
    """
    Store data calculated from the uncertainty quantification and sensitivity
    analysis for a single model/feature.

    The data types can be retrieved as attributes. Additionally, DataFeature
    implements all standard dictionary methods, such as items, value, contains
    and so implemented. This means it can be indexed as a regular dictionary
    with the data type names as keys and returns the values for that data type.

    Parameters
    ----------
    name : str
        Name of the model/feature.
    values : {None, array_like}, optional.
        Feature or model result.
        Default is None.
    time : {None, array_like}, optional.
        Time values for feature or model.
        Default is None.
    mean : {None, array_like}, optional.
        Mean of the feature or model results.
        Default is None.
    variance : {None, array_like}, optional.
        Variance of the feature or model results.
        Default is None.
    percentile_5 : {None, array_like}, optional.
        5 percentile of the feature or model results.
        Default is None.
    percentile_95 : {None, array_like}, optional.
        95 percentile of the feature or model results.
        Default is None.
    sensitivity_1 : {None, array_like}, optional.
        First order sensitivity of the feature or model results.
        Default is None.
    sensitivity_1_sum : {None, array_like}, optional.
        First order sensitivity of the feature or model results.
        Default is None.
    sensitivity_t : {None, array_like}, optional.
        Total effect sensitivity of the feature or model results.
        Default is None.
    sensitivity_t_sum : {None, array_like}, optional.
        Normalized sum of total effect sensitivity of
        the feature or model results.
        Default is None.
    labels : list, optional.
        A list of labels for plotting, ``[x-axis, y-axis, z-axis]``
        Default is ``[]``.

    Attributes
    ----------
    name : str
        Name of the model/feature.
    values : {None, array_like}
        Feature or model result.
    time : {None, array_like}
        Time values for feature or model.
    mean : {None, array_like}
        Mean of the feature or model results.
    variance : {None, array_like}
        Variance of the feature or model results.
    percentile_5 : {None, array_like}
        5 percentile of the feature or model results.
    percentile_95 : {None, array_like}
        95 percentile of the feature or model results.
    sensitivity_1 : {None, array_like}
        First order sensitivity of the feature or model results.
    sensitivity_1_sum : {None, array_like}
        First order sensitivity of the feature or model results.
    sensitivity_t : {None, array_like}
        Total effect sensitivity of the feature or model results.
    sensitivity_t_sum : {None, array_like}
        Normalized sum of total effect sensitivity of
        the feature or model results.
    labels : list
        A list of labels for plotting, ``[x-axis, y-axis, z-axis]``
    """
    def __init__(self,
                 name,
                 values=None,
                 time=None,
                 mean=None,
                 variance=None,
                 percentile_5=None,
                 percentile_95=None,
                 sensitivity_1=None,
                 sensitivity_1_sum=None,
                 sensitivity_t=None,
                 sensitivity_t_sum=None,
                 labels=[]):

        self.name = name
        self.values = values
        self.time = time
        self.mean = mean
        self.variance = variance
        self.percentile_5 = percentile_5
        self.percentile_95 = percentile_95
        self.sensitivity_1 = sensitivity_1
        self.sensitivity_1_sum = sensitivity_1_sum
        self.sensitivity_t = sensitivity_t
        self.sensitivity_t_sum = sensitivity_t_sum
        self.labels = labels

        self._built_in_data_types = ["values", "time", "mean", "variance",
                                     "percentile_5", "percentile_95",
                                     "sensitivity_1", "sensitivity_1_sum",
                                     "sensitivity_t", "sensitivity_t_sum"]

        self._information = ["name", "labels"]

    def __getitem__(self, data_type):
        """
        Get the data for `data_type`.

        Parameters
        ----------
        data_type: str
            Name of data_type.

        Returns
        -------
        {array_like, None}
            The data for `data_type`.
        """
        return getattr(self, data_type)


    def get_data_types(self):
        """
        Get the all data types that contain data (not None).

        Returns
        -------
        list
           List of all data types that contain data.
        """
        data_types = []

        for data_type in dir(self):
            if not data_type.startswith('_') and not callable(self[data_type]) \
                and self[data_type] is not None and data_type not in self._information:
                data_types.append(data_type)

        return data_types


    def __setitem__(self, data_type, data):
        """
        Set the data for `data_type`.

        Parameters
        ----------
        data_type: str
            Name of data_type.
        data : {array_like, None}
            The data for `data_type`.
        """
        setattr(self, data_type, data)


    def __iter__(self):
        """
        Iterate over each data type with data.

        Yields
        ------
        str
            Name of data type.
        """
        for data_type in self.get_data_types():
            yield data_type



    def __delitem__(self, data_type):
        """
        Delete data for `data_type` (set to None).

        Parameters
        ----------
        data_type: str
            Name of data type.
        """
        setattr(self, data_type, None)


    def __len__(self):
        """
        Get the number of data types with data.

        Returns
        -------
        int
            The number of data types with data.
        """
        return len(self.get_data_types())


    def __contains__(self, data_type):
        """
        Check if `data_type` exists and contains data (not None).

        Parameters
        ----------
        data_type: str
            Name of data type.

        Returns
        -------
        bool
            If `data_type` exists and contains data (not None)
        """
        if data_type not in self.get_data_types() or self[data_type] is None:
            return False
        else:
            return True


    def ndim(self):
        """
        Get the number of dimensions the data of a data type.

        Parameters
        ----------
        feature : str
            Name of the model or a feature.

        Returns
        -------
        int
            The number of dimensions of the data of the data type.
        """

        if self.values is not None:
            return np.ndim(self.values[0])
        else:
            return None


class Data(collections.MutableMapping):
    """
    Store data calculated from the uncertainty quantification and sensitivity
    analysis for each model/features.

    Has all standard dictionary methods, such as items, value, contains
    and so implemented. Can be indexed as a regular dictionary with
    model/feature names as keys and returns a DataFeature object that contains
    the data for all data types for that model/feature.

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
        A dictionary with a DataFeature for each model/feature.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.
    data_information : list
        List of attributes containing additional information.

    See also
    --------
    uncertainpy.DataFeature
    """
    def __init__(self,
                 filename=None,
                 verbose_level="info",
                 verbose_filename=None):

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
            If no labels are defined (labels = []),
            returns a list with the correct number of empty strings.
        """
        if self[feature].labels != []:
            return self[feature].labels

        elif self[self.model_name].labels != [] and self[self.model_name].ndim() == self[feature].ndim():
            return self[self.model_name].labels

        else:
            return [""]*(self[feature].ndim() + 1)



    def __getitem__(self, feature):
        """
        Get the DataFeature containing the data for `feature`.

        Parameters
        ----------
        feature: str
            Name of feature/model.

        Returns
        -------
        DataFeature
            The DataFeature containing the data for `feature`.
        """
        return self.data[feature]

    def __setitem__(self, feature, data):
        """
        Set `data` for `feature`. `Data` must be a DataFeature object.

        Parameters
        ----------
        feature: str
            Name of feature/model.
        data : DataFeature
            DataFeature with the data for `feature`.

        Raises
        ------
        ValueError
            If `data` is not a DataFeature.
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
        -------
        int
            The number of model/features.
        """
        return len(self.data)


    def add_features(self, features):
        """
        Add features (which contain no data).

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

