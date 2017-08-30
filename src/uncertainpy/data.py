import os
import h5py
import collections

import numpy as np

from uncertainpy.utils import create_logger


# TODO instead of a data object, could just a  h5py file have been used ?

class Data(collections.MutableMapping):
    def __init__(self,
                 filename=None,
                 verbose_level="info",
                 verbose_filename=None):

        """
        Store data calculated in the uncertainty quantification.





        Has all standard dictionary methods, such as items, value contains and so implemented.

        Parameters
        ----------
        filename : str
            Name of the file to save/load data from/to.

        Attributes
        ----------
        model_name : str
            Name of the model.

        xlabel : str
        ylabel : str

        features_0d : list
        features_1d : list
        features_2d : list
        feature_list : list

        Each feature and the model has the following data

        U : array_like
            Feature or model result.
        t : array_like
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
            Normalized sum of total effect sensitivity of the feature or model results.

        """


        self.data_types = ["U", "t", "E", "Var", "p_05", "p_95",
                           "sensitivity_1", "total_sensitivity_1",
                           "sensitivity_t", "total_sensitivity_t", "labels"]


        self.data_information = ["uncertain_parameters", "model_name"]

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


        self.uncertain_parameters = []
        self.model_name = ""
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
            for data_type in self[feature]:
                output_str += "=== {data_type} ===\n".format(data_type=data_type)
                output_str += "{data}\n\n".format(data=self[feature][data_type])


        return output_str.strip()



    def clear(self):
        self.uncertain_parameters = []
        self.model_name = ""

        self.data = {}


    def ndim(self, feature):
    #     if "U" in self[feature]:
    #         ndim = np.ndim(self[feature]["U"])
        return np.ndim(self[feature]["U"][0])



    def get_labels(self, feature):
        if "labels" in self[feature]:
            return self[feature]["labels"]

        elif self.ndim(feature) == 2:
            if self.ndim(self.model_name) == 2 and "labels" in self[self.model_name]:
                return self[self.model_name]["labels"]
            else:
                return ["", "", ""]

        elif self.ndim(feature) == 1:
            if self.ndim(self.model_name) == 1 and "labels" in self[self.model_name]:
                return self[self.model_name]["labels"]
            else:
                return ["", ""]

        elif self.ndim(feature) == 0:
            if self.ndim(self.model_name) == 0 and "labels" in self[self.model_name]:
                return self[self.model_name]["labels"]
            else:
                return [""]


    def __getitem__(self, feature):
        return self.data[feature]

    def __setitem__(self, feature, value):
        if not isinstance(value, dict):
            raise ValueError("Value must be of type dict")
        self.data[feature] = value


    def __iter__(self):
        return iter(self.data)


    def __delitem__(self, feature):
        del self.data[feature]


    def __len__(self):
        return len(self.data)


    def add_features(self, features):
        if isinstance(features, str):
            features = [features]

        for feature in features:
            self.data[feature] = {}


    def is_adaptive(self, feature):
        """
Test if the model returned an adaptive result
        """
        u_prev = self[feature]["U"][0]
        for u in self[feature]["U"][1:]:
            if u_prev.shape != u.shape:
                return True
            u_prev = u
        return False


    def save(self, filename):
        ### TODO expand the save function to also save parameters and model information

        with h5py.File(os.path.join(filename), 'w') as f:
            f.attrs["uncertain parameters"] = self.uncertain_parameters
            f.attrs["model name"] = self.model_name

            for feature in self:
                group = f.create_group(feature)

                for data_type in self[feature]:
                    group.create_dataset(data_type, data=self[feature][data_type])


    def load(self, filename):
        self.filename = filename

        # TODO add this check when changing to python 3
        # if not os.path.isfile(self.filename):
        #     raise FileNotFoundError("{} file not found".format(self.filename))
        self.clear()

        with h5py.File(self.filename, 'r') as f:
            self.uncertain_parameters = list(f.attrs["uncertain parameters"])
            self.model_name = f.attrs["model name"]

            for feature in f:
                self.add_features(str(feature))
                for data_type in f[feature]:
                    self[feature][data_type] = f[feature][data_type][()]



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


    def remove_only_invalid_results(self):
        feature_list = self.data.keys()[:]
        for feature in feature_list:
            all_nan = True
            for U in self[feature]["U"]:
                if not np.all(np.isnan(U)):
                    all_nan = False

            if all_nan:
                self.logger.warning("Feature: {} does".format(feature)
                                    + " not yield results for any parameter combinations")

                del self[feature]
