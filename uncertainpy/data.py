import os
import h5py

from uncertainpy.utils import create_logger



class Data:
    def __init__(self,
                 filename=None,
                 verbose_level="info",
                 verbose_filename=None):

        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


        if filename is None:
            self.resetValues()
        else:
            self.load(filename)


    def resetValues(self):
        self.uncertain_parameters = None

        self.features_0d = []
        self.features_1d = []
        self.features_2d = []
        self.feature_list = []


        self.U = {}
        self.t = {}
        self.E = {}
        self.Var = {}
        self.p_05 = {}
        self.p_95 = {}
        self.sensitivity = {}

        self.xlabel = ""
        self.ylabel = ""



    def save(self, filename):
        ### TODO expand the save funcition to also save parameters and model information

        with h5py.File(os.path.join(filename), 'w') as f:

            # f.attrs["name"] = self.output_file.split("/")[-1]
            f.attrs["uncertain parameters"] = self.uncertain_parameters
            f.attrs["features"] = self.feature_list
            f.attrs["xlabel"] = self.xlabel
            f.attrs["ylabel"] = self.ylabel

            for feature in self.feature_list:
                group = f.create_group(feature)

                if feature in self.t and self.t[feature] is not None:
                    group.create_dataset("t", data=self.t[feature])
                # IMPROVEMENT do not save U to save space?
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
                if feature in self.sensitivity and self.sensitivity[feature] is not None:
                    group.create_dataset("sensitivity", data=self.sensitivity[feature])
                # if feature == "directComparison":
                #    group.create_dataset("total sensitivity", data=self.sensitivity_ranking[parameter])


    def load(self, filename):
        self.filename = filename

        with h5py.File(self.filename, 'r') as f:
            self.t = {}
            self.U = {}
            self.E = {}
            self.Var = {}
            self.p_05 = {}
            self.p_95 = {}
            self.sensitivity = {}

            for feature in f.keys():
                self.U[feature] = f[feature]["U"][()]
                self.E[feature] = f[feature]["E"][()]
                self.Var[feature] = f[feature]["Var"][()]
                self.p_05[feature] = f[feature]["p_05"][()]
                self.p_95[feature] = f[feature]["p_95"][()]

                if "sensitivity" in f[feature].keys():
                    self.sensitivity[feature] = f[feature]["sensitivity"][()]
                else:
                    self.sensitivity[feature] = None

                if "t" in f[feature].keys():
                    self.t[feature] = f[feature]["t"][()]
                # else:
                #     self.t[feature] = None

            self.setFeatures(self.E)
            self.uncertain_parameters = f.attrs["uncertain parameters"]

            self.xlabel = f.attrs["xlabel"]
            self.ylabel = f.attrs["ylabel"]


    def setFeatures(self, results):
        self.features_0d, self.features_1d, self.features_2d = self.sortFeatures(results)
        self.feature_list = self.features_0d + self.features_1d + self.features_2d


    def sortFeatures(self, results):
        features_2d = []
        features_1d = []
        features_0d = []

        for feature in results:
            if hasattr(results[feature], "__iter__"):
                if len(results[feature].shape) == 0:
                    features_0d.append(feature)
                elif len(results[feature].shape) == 1:
                    features_1d.append(feature)
                else:
                    features_2d.append(feature)
            else:
                features_0d.append(feature)

        return features_0d, features_1d, features_2d
