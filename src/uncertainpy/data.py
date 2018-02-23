import os
import h5py
import collections

import numpy as np

from .utils import create_logger
from ._version import __version__


class DataFeature(collections.MutableMapping):
    """
    Store the results of each statistical metric calculated from the uncertainty
    quantification and sensitivity analysis for a single model/feature.

    The statistical metrics can be retrieved as attributes. Additionally, DataFeature
    implements all standard dictionary methods, such as items, value, contains
    and so implemented. This means it can be indexed as a regular dictionary
    with the statistical metric names as keys and returns the values for that
    statistical metric.

    Parameters
    ----------
    name : str
        Name of the model/feature.
    evaluations : {None, array_like}, optional.
        Feature or model result.
        Default is None.
    time : {None, array_like}, optional.
        Time evaluations for feature or model.
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
    sobol_first : {None, array_like}, optional.
        First order sensitivity of the feature or model results.
        Default is None.
    sobol_first_sum : {None, array_like}, optional.
        First order sensitivity of the feature or model results.
        Default is None.
    sobol_total : {None, array_like}, optional.
        Total effect sensitivity of the feature or model results.
        Default is None.
    sobol_total_sum : {None, array_like}, optional.
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
    evaluations : {None, array_like}
        Feature or model output.
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
    sobol_first : {None, array_like}
        First order sensitivity of the feature or model results.
    sobol_first_sum : {None, array_like}
        First order sensitivity of the feature or model results.
    sobol_total : {None, array_like}
        Total effect sensitivity of the feature or model results.
    sobol_total_sum : {None, array_like}
        Normalized sum of total effect sensitivity of
        the feature or model results.
    labels : list
        A list of labels for plotting, ``[x-axis, y-axis, z-axis]``.

    Notes
    -----
    The statistical metrics calculated in Uncertainpy are:

        * ``evaluations`` - the results from the model/feature evaluations.
        * ``time`` - the time of the model/feature.
        * ``mean`` - the mean of the model/feature.
        * ``variance``. - the variance of the model/feature.
        * ``percentile_5`` - the 5th percentile of the model/feature.
        * ``percentile_95`` - the 95th percentile of the model/feature.
        * ``sobol_first`` - the first order Sobol indices (sensitivity) of
          the model/feature.
        * ``sobol_first_sum`` - the total order Sobol indices (sensitivity)
          of the model/feature.
        * ``sobol_total`` - the normalized sum of the first order Sobol
          indices (sensitivity) of the model/feature.
        * ``sobol_total_sum`` - the normalized sum of the total order Sobol
          indices (sensitivity) of the model/feature.
    """
    def __init__(self,
                 name,
                 evaluations=None,
                 time=None,
                 mean=None,
                 variance=None,
                 percentile_5=None,
                 percentile_95=None,
                 sobol_first=None,
                 sobol_first_sum=None,
                 sobol_total=None,
                 sobol_total_sum=None,
                 labels=[]):

        self.name = name
        self.evaluations = evaluations
        self.time = time
        self.mean = mean
        self.variance = variance
        self.percentile_5 = percentile_5
        self.percentile_95 = percentile_95
        self.sobol_first = sobol_first
        self.sobol_first_sum = sobol_first_sum
        self.sobol_total = sobol_total
        self.sobol_total_sum = sobol_total_sum
        self.labels = labels

        self._statistical_metrics = ["evaluations", "time", "mean", "variance",
                                     "percentile_5", "percentile_95",
                                     "sobol_first", "sobol_first_sum",
                                     "sobol_total", "sobol_total_sum"]

        self._information = ["name", "labels"]

    def __getitem__(self, statistical_metric):
        """
        Get the data for `statistical_metric`.

        Parameters
        ----------
        statistical_metric: str
            Name of the statistical metric.

        Returns
        -------
        {array_like, None}
            The data for `statistical_metric`.
        """
        return getattr(self, statistical_metric)


    def get_metrics(self):
        """
        Get the names of all statistical metrics that contain data (not None).

        Returns
        -------
        list
           List of the names of all statistical metric that contain data.
        """
        statistical_metrics = []

        for statistical_metric in dir(self):
            if not statistical_metric.startswith('_') and not callable(self[statistical_metric]) \
                and self[statistical_metric] is not None and statistical_metric not in self._information:
                statistical_metrics.append(statistical_metric)

        return statistical_metrics


    def __setitem__(self, statistical_metric, data):
        """
        Set the data for the statistical metric.

        Parameters
        ----------
        statistical_metric: str
            Name of the statistical metric.
        data : {array_like, None}
            The data for the statistical metric.
        """
        setattr(self, statistical_metric, data)


    def __iter__(self):
        """
        Iterate over each statistical metric with data.

        Yields
        ------
        str
            Name of the statistical metric.
        """
        for statistical_metric in self.get_metrics():
            yield statistical_metric



    def __delitem__(self, statistical_metric):
        """
        Delete data for `statistical_metric` (set to None).

        Parameters
        ----------
        statistical_metric: str
            Name of the statistical metric.
        """
        setattr(self, statistical_metric, None)


    def __len__(self):
        """
        Get the number of data types with data.

        Returns
        -------
        int
            The number of data types with data.
        """
        return len(self.get_metrics())


    def __contains__(self, statistical_metric):
        """
        Check if `statistical_metric` exists and contains data (not None).

        Parameters
        ----------
        statistical_metric: str
            Name of the statistical metric.

        Returns
        -------
        bool
            If `statistical_metric` exists and contains data (not None)
        """
        if statistical_metric not in self.get_metrics() or self[statistical_metric] is None:
            return False
        else:
            return True


    def __str__(self):
        """
        Convert all data to a readable string.

        Returns
        -------
        str
           A human readable string of all statistical metrics.
        """
        output_str = ""
        for statistical_metric in self:
            output_str += "=== {statistical_metric} ===\n".format(statistical_metric=statistical_metric)
            output_str += "{data}\n\n".format(data=self[statistical_metric])


        return output_str.strip()

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

        if self.evaluations is not None:
            return np.ndim(self.evaluations[0])
        else:
            return None


class Data(collections.MutableMapping):
    """
    Store the results of each statistical metric calculated from the uncertainty
    quantification and sensitivity analysis for each model/features.

    Has all standard dictionary methods, such as items, value, contains
    and so implemented. Can be indexed as a regular dictionary with
    model/feature names as keys and returns a DataFeature object that contains
    the data for all statistical metrics for that model/feature.
    Additionally it contains information on how the calculations was performed

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
    uncertain_parameters : list
        A list of the uncertain parameters in the uncertainty quantification.
    model_name : str
        Name of the model.
    incomplete : list
        List of all model/features that have missing model/feature evaluations.
    method : str
        A string that describes the method used to perform the uncertainty
        quantification.
    data : dictionary
        A dictionary with a DataFeature for each model/feature.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.
    data_information : list
        List of attributes containing additional information.


    Notes
    -----
    The statistical metrics calculated for each feature and model in Uncertainpy
    are:

        * ``evaluations`` - the results from the model/feature evaluations.
        * ``time`` - the time of the model/feature.
        * ``mean`` - the mean of the model/feature.
        * ``variance``. - the variance of the model/feature.
        * ``percentile_5`` - the 5th percentile of the model/feature.
        * ``percentile_95`` - the 95th percentile of the model/feature.
        * ``sobol_first`` - the first order Sobol indices (sensitivity) of
          the model/feature.
        * ``sobol_first_sum`` - the total order Sobol indices (sensitivity)
          of the model/feature.
        * ``sobol_total`` - the normalized sum of the first order Sobol
          indices (sensitivity) of the model/feature.
        * ``sobol_total_sum`` - the normalized sum of the total order Sobol
          indices (sensitivity) of the model/feature.

    See also
    --------
    uncertainpy.DataFeature
    """
    def __init__(self,
                 filename=None,
                 verbose_level="info",
                 verbose_filename=None):

        self.data_information = ["uncertain_parameters", "model_name",
                                 "incomplete", "method", "version", "seed"]

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


        self.uncertain_parameters = []
        self.model_name = ""
        self.incomplete = []
        self.data = {}
        self.method = ""
        self._seed = ""

        self.version = __version__

        if filename is not None:
            self.load(filename)


    @property
    def seed(self):
        """
        Seed used in the calculations.

        Parameters
        ----------
        new_seed : {None, int}
            Seed used in the calculations.
            If None, converted to "".

        Returns
        -------
        seed : {int, str}
            Seed used in the calculations.
        """
        return self._seed


    @seed.setter
    def seed(self, new_seed):
        if new_seed is None:
            self._seed = ""
        else:
            self._seed = new_seed


    def __str__(self):
        """
        Convert all data to a readable string.

        Returns
        -------
        str
           A human readable string of all stored data.
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
            output_str += str(self[feature]) + "\n"

        return output_str.strip()



    def clear(self):
        """
        Clear all data.
        """
        self.uncertain_parameters = []
        self.model_name = ""
        self.incomplete = []
        self.data = {}
        self.method = ""
        self._seed = ""

        self.version = __version__


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


    # TODO expand the save function to also save parameters and model information
    def save(self, filename):
        """
        Save data to a hdf5 file with name `filename`.

        Parameters
        ----------
        filename : str
            Name of the file to load data from.
        """
        with h5py.File(filename, 'w') as f:
            f.attrs["uncertain parameters"] = self.uncertain_parameters
            f.attrs["model name"] = self.model_name
            f.attrs["incomplete results"] = self.incomplete
            f.attrs["method"] = self.method
            f.attrs["version"] = self.version
            f.attrs["seed"] = self.seed


            for feature in self:
                group = f.create_group(feature)

                for statistical_metric in self[feature]:
                    group.create_dataset(statistical_metric, data=self[feature][statistical_metric])

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
            self.incomplete = list(f.attrs["incomplete results"])
            self.method = f.attrs["method"]
            self.version = f.attrs["version"]
            self.seed = f.attrs["seed"]

            for feature in f:
                self.add_features(str(feature))
                for statistical_metric in f[feature]:
                    self[feature][statistical_metric] = f[feature][statistical_metric][()]


    def remove_only_invalid_features(self):
        """
        Remove all features that only have invalid results (NaN).
        """
        feature_list = self.data.keys()[:]
        for feature in feature_list:
            all_nan = True
            for U in self[feature].evaluations:
                if not np.all(np.isnan(U)):
                    all_nan = False

            if all_nan:
                self.logger.warning("Feature: {} does".format(feature)
                                    + " not yield results for any parameter combinations")

                del self[feature]

