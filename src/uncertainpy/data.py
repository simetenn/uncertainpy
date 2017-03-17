import os
import h5py

import numpy as np

from uncertainpy.utils import create_logger



class Data:
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


        self.xlabel = ""
        self.ylabel = ""


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
                    group.create_dataset("t", data=self.t[feature])
                if feature in self.U:
                    group.create_dataset("U", data=self.U[feature])
                if feature in self.E:
                    group.create_dataset("E", data=self.E[feature])
                if feature in self.Var:
                    group.create_dataset("Var", data=self.Var[feature])
                if feature in self.p_05:
                    group.create_dataset("p_05", data=self.p_05[feature])
                if feature in self.p_95:
                    group.create_dataset("p_95", data=self.p_95[feature])
                if feature in self.sensitivity_1 and self.sensitivity_1[feature] is not None:
                    group.create_dataset("sensitivity_1", data=self.sensitivity_1[feature])
                if feature in self.total_sensitivity_1 and self.total_sensitivity_1[feature] is not None:
                    group.create_dataset("total_sensitivity_1", data=self.total_sensitivity_1[feature])
                if feature in self.sensitivity_t and self.sensitivity_t[feature] is not None:
                    group.create_dataset("sensitivity_t", data=self.sensitivity_t[feature])
                if feature in self.total_sensitivity_t and self.total_sensitivity_t[feature] is not None:
                    group.create_dataset("total_sensitivity_t", data=self.total_sensitivity_t[feature])


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

            self.feature_list = f.attrs["features"]
            self.features_0d = f.attrs["features_0d"]
            self.features_1d = f.attrs["features_1d"]
            self.features_2d = f.attrs["features_2d"]


            for feature in f.keys():
                self.U[feature] = f[feature]["U"][()]
                self.E[feature] = f[feature]["E"][()]
                self.Var[feature] = f[feature]["Var"][()]
                self.p_05[feature] = f[feature]["p_05"][()]
                self.p_95[feature] = f[feature]["p_95"][()]


                if "sensitivity_1" in f[feature].keys():
                    self.sensitivity_1[feature] = f[feature]["sensitivity_1"][()]
                else:
                    self.sensitivity_1[feature] = None

                if "total_sensitivity_1" in f[feature].keys():
                    self.total_sensitivity_1[feature] = f[feature]["total_sensitivity_1"][()]
                else:
                    self.total_sensitivity_1[feature] = None



                if "sensitivity_t" in f[feature].keys():
                    self.sensitivity_t[feature] = f[feature]["sensitivity_t"][()]
                else:
                    self.sensitivity_t[feature] = None

                if "total_sensitivity_t" in f[feature].keys():
                    self.total_sensitivity_t[feature] = f[feature]["total_sensitivity_t"][()]
                else:
                    self.total_sensitivity_t[feature] = None


                if "t" in f[feature].keys():
                    self.t[feature] = f[feature]["t"][()]
                else:
                    self.t[feature] = None

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
    def features_1d(self, new_features_2d):
        self._features_2d = new_features_2d
        self._update_feature_list()



    def _update_feature_list(self):
        self.feature_list = self._features_0d + self._features_1d + self._features_2d
        self.feature_list.sort()


    # def setFeatures(self, results):
    #     self.features_0d, self.features_1d, self.features_2d = self.sortFeatures(results)
    #     self.feature_list = self.features_0d + self.features_1d + self.features_2d
    #     self.feature_list.sort()


    # def sortFeatures(self, results):
    #
    #     """
    #     results = {'directComparison': (None,
    #                                     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #                                     None),
    #                'feature2d': (None,
    #                              array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    #                              None),
    #                'feature1d': (None,
    #                              array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #                              None),
    #                'feature0d': (None,
    #                              1,
    #                              None)}
    #     """
    #
    #     features_2d = []
    #     features_1d = []
    #     features_0d = []
    #
    #     for feature in results:
    #         if hasattr(results[feature][1], "__iter__"):
    #
    #             if len(results[feature][1].shape) == 0:
    #                 features_0d.append(feature)
    #             elif len(results[feature][1].shape) == 1:
    #                 features_1d.append(feature)
    #             else:
    #                 features_2d.append(feature)
    #         else:
    #             features_0d.append(feature)
    #
    #     return features_0d, features_1d, features_2d



    def removeOnlyInvalidResults(self):
        old_feature_list = self.feature_list[:]
        for feature in old_feature_list:
            if np.all(np.isnan(self.U[feature])):
                self.logger.warning("Feature: {} does not yield results for any parameter combinations".format(feature))
                # raise RuntimeWarning("Feature: {} does not yield results for any parameter combinations".format(feature))

                self.U[feature] = "Only invalid results for all set of parameters"
                self.feature_list.remove(feature)

                if feature in self.features_0d:
                    self.features_0d.remove(feature)

                if feature in self.features_2d:
                    self.features_2d.remove(feature)

                if feature in self.features_1d:
                    self.features_1d.remove(feature)
