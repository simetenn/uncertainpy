import os
import h5py

import numpy as np

from uncertainpy.utils import create_logger



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

        self.data_names = ["U", "t", "E", "Var", "p_05", "p_95",
                           "sensitivity_1", "total_sensitivity_1",
                           "sensitivity_t", "total_sensitivity_t"]


        self.data_information = ["xlabel", "ylabel", "features_0d",
                                 "features_1d", "features_2d", "feature_list"]

        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


        self.resetValues()

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



    def resetValues(self):
        self.uncertain_parameters = None

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



    def isAdaptive(self):
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
        ### TODO expand the save funcition to also save parameters and model information

        with h5py.File(os.path.join(filename), 'w') as f:

            # f.attrs["name"] = self.output_file.split("/")[-1]
            f.attrs["uncertain parameters"] = self.uncertain_parameters
            f.attrs["features"] = self.feature_list
            f.attrs["xlabel"] = self.xlabel
            f.attrs["ylabel"] = self.ylabel
            f.attrs["features_0d"] = self.features_0d
            f.attrs["features_1d"] = self.features_1d
            f.attrs["features_2d"] = self.features_2d


            for feature in self.feature_list:
                group = f.create_group(feature)

                if feature in self.t and self.t[feature] is not None:
                    group.create_dataset("t", data=self.none_to_nan(self.t[feature]))
                if feature in self.U:
                    group.create_dataset("U", data=self.none_to_nan(self.U[feature]))
                if feature in self.E:
                    group.create_dataset("E", data=self.none_to_nan(self.E[feature]))
                if feature in self.Var:
                    group.create_dataset("Var", data=self.none_to_nan(self.Var[feature]))
                if feature in self.p_05:
                    group.create_dataset("p_05", data=self.none_to_nan(self.p_05[feature]))
                if feature in self.p_95:
                    group.create_dataset("p_95", data=self.none_to_nan(self.p_95[feature]))
                if feature in self.sensitivity_1 and self.sensitivity_1[feature] is not None:
                    group.create_dataset("sensitivity_1", data=self.none_to_nan(self.sensitivity_1[feature]))
                if feature in self.total_sensitivity_1 and self.total_sensitivity_1[feature] is not None:
                    group.create_dataset("total_sensitivity_1", data=self.none_to_nan(self.total_sensitivity_1[feature]))
                if feature in self.sensitivity_t and self.sensitivity_t[feature] is not None:
                    group.create_dataset("sensitivity_t", data=self.none_to_nan(self.sensitivity_t[feature]))
                if feature in self.total_sensitivity_t and self.total_sensitivity_t[feature] is not None:
                    group.create_dataset("total_sensitivity_t", data=self.none_to_nan(self.total_sensitivity_t[feature]))


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


            self.uncertain_parameters = f.attrs["uncertain parameters"]

            self.xlabel = f.attrs["xlabel"]
            self.ylabel = f.attrs["ylabel"]


            # self.feature_list = listf.attrs["features"]
            self.features_0d = list(f.attrs["features_0d"])
            self.features_1d = list(f.attrs["features_1d"])
            self.features_2d = list(f.attrs["features_2d"])


            for feature in f.keys():
                self.U[feature] = self.nan_to_none(f[feature]["U"][()])
                self.E[feature] = self.nan_to_none(f[feature]["E"][()])
                self.Var[feature] = self.nan_to_none(f[feature]["Var"][()])
                self.p_05[feature] = self.nan_to_none(f[feature]["p_05"][()])
                self.p_95[feature] = self.nan_to_none(f[feature]["p_95"][()])


                if "sensitivity_1" in f[feature].keys():
                    self.sensitivity_1[feature] = self.nan_to_none(f[feature]["sensitivity_1"][()])
                else:
                    self.sensitivity_1[feature] = None

                if "total_sensitivity_1" in f[feature].keys():
                    self.total_sensitivity_1[feature] = self.nan_to_none(f[feature]["total_sensitivity_1"][()])
                else:
                    self.total_sensitivity_1[feature] = None



                if "sensitivity_t" in f[feature].keys():
                    self.sensitivity_t[feature] = self.nan_to_none(f[feature]["sensitivity_t"][()])
                else:
                    self.sensitivity_t[feature] = None

                if "total_sensitivity_t" in f[feature].keys():
                    self.total_sensitivity_t[feature] = self.nan_to_none(f[feature]["total_sensitivity_t"][()])
                else:
                    self.total_sensitivity_t[feature] = None


                if "t" in f[feature].keys():
                    self.t[feature] = f[feature]["t"][()]
                else:
                    self.t[feature] = None

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


    def removeOnlyInvalidResults(self):
        old_feature_list = self.feature_list[:]
        for feature in old_feature_list:

            all_none = True
            for U in self.U[feature]:
                if U is not None:
                    all_none = False

            if all_none:
                self.logger.warning("Feature: {} does".format(feature)
                                    + " not yield results for any parameter combinations")
                # raise RuntimeWarning("Feature: {} does not yield results for any parameter combinations".format(feature))

                self.U[feature] = "Only invalid results for all set of parameters"
                self.feature_list.remove(feature)

                if feature in self.features_0d:
                    self.features_0d.remove(feature)

                if feature in self.features_2d:
                    self.features_2d.remove(feature)

                if feature in self.features_1d:
                    self.features_1d.remove(feature)
