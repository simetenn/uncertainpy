import os
import h5py

import numpy as np

from uncertainpy.utils import create_logger


# TODO instead of a data object, could just a  h5py file have been used ?

class Data(object):
    def __init__(self,
                 filename=None,
                 verbose_level="info",
                 verbose_filename=None):

        """
U
t
E
Var
p_05
p_95
sensitivity_1
total_sensitivity_1
sensitivity_t
total_sensitivity_t

xlabel
ylabel

features_0d
features_1d
features_2d
feature_list
        """

        self.data_types = ["U", "t", "E", "Var", "p_05", "p_95",
                           "sensitivity_1", "total_sensitivity_1",
                           "sensitivity_t", "total_sensitivity_t", "labels"]


        self.data_information = ["features_0d", "features_1d", "features_2d",
                                 "uncertain_parameters", "model_name"]

        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        self.clear()

        if filename is not None:
            self.load(filename)


    def __str__(self):
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

        output_str += border("Content")



        for feature in self:
            output_str += border(feature)
            for data_type in self[feature]:
                output_str += "=== {data_type} ===\n".format(data_type=data_type)
                output_str += "{data}\n\n".format(data=self[feature][data_type])


        return output_str.strip()



    def clear(self):
        self.uncertain_parameters = []

        self.features_0d = []
        self.features_1d = []
        self.features_2d = []

        self.data = {}

        self.model_name = ""


    # @property
    # def features_0d(self):
    #     return self._features_0d

    # @features_0d.setter
    # def features_0d(self, new_features_0d):
    #     self._features_0d = new_features_0d
    #     self._update_feature_list()

    # @property
    # def features_1d(self):
    #     return self._features_1d

    # @features_1d.setter
    # def features_1d(self, new_features_1d):
    #     self._features_1d = new_features_1d
    #     self._update_feature_list()

    # @property
    # def features_2d(self):
    #     return self._features_2d

    # @features_2d.setter
    # def features_2d(self, new_features_2d):
    #     self._features_2d = new_features_2d
    #     self._update_feature_list()

    # def _update_feature_list(self):
    #     self.feature_list = self._features_0d + self._features_1d + self._features_2d
    #     self.feature_list.sort()


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


    def get_labels(self, feature):
        if "labels" in self[feature]:
            return self[feature]["labels"]

        elif feature in self.features_2d:
            if self.model_name in self.features_2d and "labels" in self[self.model_name]:
                return self[self.model_name]["labels"]
            else:
                return ["", "", ""]
        elif feature in self.features_1d:
            if self.model_name in self.features_1d and "labels" in self[self.model_name]:
                return self[self.model_name]["labels"]
            else:
                return ["", ""]
        elif feature in self.features_0d:
            if self.model_name in self.features_0d and "labels" in self[self.model_name]:
                return self[self.model_name]["labels"]
            else:
                return [""]


    def __getitem__(self, feature):
        return self.data[feature]


    def __iter__(self):
        return iter(self.data)


    def __contains__(self, feature):
        return feature in self.data


    def __delitem__(self, feature):
        del self.data[feature]


    def add_features(self, features):
        if isinstance(features, str):
            features = [features]

        for feature in features:
            self.data[feature] = {}

            # for data_type in self.data_types:
            #     self.data[feature][data_type] = None



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
                # raise RuntimeWarning("Feature: {} does not yield
                # results for any parameter combinations".format(feature))

                del self[feature]

                if feature in self.features_0d:
                    self.features_0d.remove(feature)

                if feature in self.features_2d:
                    self.features_2d.remove(feature)

                if feature in self.features_1d:
                    self.features_1d.remove(feature)
