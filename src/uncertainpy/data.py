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

        # TODO consider storing all data belonging to one
        # specific feature in a dict for that feature

        self.data_names = ["U", "t", "E", "Var", "p_05", "p_95",
                           "sensitivity_1", "total_sensitivity_1",
                           "sensitivity_t", "total_sensitivity_t"]


        self.data_information = ["xlabel", "ylabel", "features_0d",
                                 "features_1d", "features_2d", "feature_list",
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

        output_str = border("Information on Data")

        for info in self.data_information:
            current_info = getattr(self, info)
            output_str += "{info}: {current_info}\n".format(info=info,
                                                            current_info=current_info)

        output_str += border("Content of Data")
        for name in self.data_names:

            output_str += border(name)
            current_data = getattr(self, name)

            for feature in self.feature_list:
                output_str += "=== {feature} ===\n".format(feature=feature)
                if feature in current_data:
                    output_str += "{data}\n\n".format(data=current_data[feature])
                else:
                    output_str += "No data\n\n"


        return output_str.strip()



    def clear(self):
        self.uncertain_parameters = []

        self._features_0d = []
        self._features_1d = []
        self._features_2d = []
        self.feature_list = []


        self.U = {}
        self.t = {}
        self.E = {}
        self.Var = {}
        self.p_05 = {}
        self.p_95 = {}
        self.sensitivity_1 = {}
        self.total_sensitivity_1 = {}
        self.sensitivity_t = {}
        self.total_sensitivity_t = {}
        self._temperature = 0

        self.xlabel = ""
        self.ylabel = ""
        self.zlabel = ""
        self.model_name = ""


    @property
    def features_0d(self):
        return self._features_0d

    @features_0d.setter
    def features_0d(self, new_features_0d):
        self._features_0d = new_features_0d
        self._update_feature_list()

    @property
    def features_1d(self):
        return self._features_1d

    @features_1d.setter
    def features_1d(self, new_features_1d):
        self._features_1d = new_features_1d
        self._update_feature_list()

    @property
    def features_2d(self):
        return self._features_2d

    @features_2d.setter
    def features_2d(self, new_features_2d):
        self._features_2d = new_features_2d
        self._update_feature_list()


    def _update_feature_list(self):
        self.feature_list = self._features_0d + self._features_1d + self._features_2d
        self.feature_list.sort()



    def is_adaptive(self):
        """
Test if the model returned an adaptive result
        """
        for feature in self.features_1d + self.features_2d:
            u_prev = self.U[feature][0]
            for u in self.U[feature][1:]:
                if u_prev.shape != u.shape:
                    return True
                u_prev = u
        return False


    def save(self, filename):
        ### TODO expand the save function to also save parameters and model information

        with h5py.File(os.path.join(filename), 'w') as f:

            # f.attrs["name"] = self.output_file.split("/")[-1]
            f.attrs["uncertain parameters"] = self.uncertain_parameters
            f.attrs["features"] = self.feature_list
            f.attrs["xlabel"] = self.xlabel
            f.attrs["ylabel"] = self.ylabel
            f.attrs["zlabel"] = self.zlabel
            f.attrs["features_0d"] = self.features_0d
            f.attrs["features_1d"] = self.features_1d
            f.attrs["features_2d"] = self.features_2d
            f.attrs["model name"] = self.model_name


            for feature in self.feature_list:
                group = f.create_group(feature)

                for data_name in self.data_names:
                    data = getattr(self, data_name)

                    if feature in data and data[feature] is not None:
                        # group.create_dataset(data_name, data=self.none_to_nan(data[feature]))
                        group.create_dataset(data_name, data=data[feature])


    def load(self, filename):
        self.filename = filename

        # TODO add this check when changing to python 3
        # if not os.path.isfile(self.filename):
        #     raise FileNotFoundError("{} file not found".format(self.filename))

        with h5py.File(self.filename, 'r') as f:
            self.t = {}
            self.U = {}
            self.E = {}
            self.Var = {}
            self.p_05 = {}
            self.p_95 = {}
            self.sensitivity_1 = {}


            self.uncertain_parameters = list(f.attrs["uncertain parameters"])

            self.xlabel = f.attrs["xlabel"]
            self.ylabel = f.attrs["ylabel"]
            self.zlabel = f.attrs["zlabel"]
            self.model_name = f.attrs["model name"]


            # self.feature_list = listf.attrs["features"]
            self.features_0d = list(f.attrs["features_0d"])
            self.features_1d = list(f.attrs["features_1d"])
            self.features_2d = list(f.attrs["features_2d"])


            for feature in f.keys():
                for data_name in self.data_names:
                    data = getattr(self, data_name)

                    if data_name in f[feature].keys():
                        # data[feature] = self.nan_to_none(f[feature][data_name][()])
                        data[feature] = f[feature][data_name][()]
                    else:
                        data[feature] = None



    # TODO rewrite so this is using np.nan. Currently this makes it so when loading a data file it is loaded as an object
    def nan_to_none(self, array):
        try:
            tmp_array = array.astype(object)
            tmp_array[np.isnan(array)] = None

            return tmp_array

        except TypeError:
            return array


    def none_to_nan(self, array):
        tmp_array = np.array(array, dtype=float)

        return tmp_array

    #
    # def all_to_none(self):
    #     for data_name in self.data_names:
    #         data = getattr(self, data_name)
    #         if data is not None:
    #             for feature in data:
    #                 print data[feature]
    #                 data[feature] = self.nan_to_none(data[feature])
    #
    #         data = self.nan_to_none(data)
    #
    # def all_to_nan(self):
    #     for data_name in self.data_names:
    #         data = getattr(self, data_name)
    #         if data is not None:
    #             for feature in data:
    #                 data[feature] = self.none_to_nan(data[feature])
    #
    #         data = self.nan_to_none(data)

    def remove_only_invalid_results(self):
        old_feature_list = self.feature_list[:]
        for feature in old_feature_list:

            all_nan = True
            for U in self.U[feature]:
                if not np.all(np.isnan(U)):
                    all_nan = False

            if all_nan:
                self.logger.warning("Feature: {} does".format(feature)
                                    + " not yield results for any parameter combinations")
                # raise RuntimeWarning("Feature: {} does not yield
                # results for any parameter combinations".format(feature))

                self.U[feature] = "Only invalid results for all set of parameters"
                self.feature_list.remove(feature)

                if feature in self.features_0d:
                    self.features_0d.remove(feature)

                if feature in self.features_2d:
                    self.features_2d.remove(feature)

                if feature in self.features_1d:
                    self.features_1d.remove(feature)
