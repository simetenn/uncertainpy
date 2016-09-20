import os
import h5py

from uncertainpy import create_logger

class Data:
    def __init__(self,
                 verbose_level="info",
                 verbose_filename=None):

        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        self.loaded_flag = False

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)



        self.resetValues()



    def resetValues(self):
        self.uncertain_parameters = None

        self.features_0d = []
        self.features_1d = []
        self.features_2d = []
        self.sfeatures = []


        self.U = {}
        self.t = {}
        self.E = {}
        self.Var = {}
        self.p_05 = {}
        self.p_95 = {}
        self.sensitivity = {}



    def save(self, filename):
        ### TODO expand the save funcition to also save parameters and model information

        f = h5py.File(os.path.join(filename), 'w')

        # f.attrs["name"] = self.output_file.split("/")[-1]
        f.attrs["uncertain parameters"] = self.uncertain_parameters
        f.attrs["features"] = self.features


        for feature in self.all_features:
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

        f.close()



    def loadData(self, filename):
        self.filename = filename

        f = h5py.File(self.filename, 'r')

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

        self.features_0d, self.features_1d, self.features_2d = self.sortFeatures(self.E)

        self.uncertain_parameters = f.attrs["uncertain parameters"]

        self.loaded_flag = True


    def setData(self, t, U, E, Var, p_05, p_95, uncertain_parameters,
                sensitivity):

        self.t = t
        self.U = U
        self.E = E
        self.Var = Var
        self.p_05 = p_05
        self.p_95 = p_95
        self.sensitivity = sensitivity
        self.uncertain_parameters = uncertain_parameters


        self.features_0d, self.features_1d, self.features_2d = self.sortFeatures(self.E)

        self.loaded_flag = True


    def sortFeatures(self, results):
        features_0d = []
        features_1d = []
        features_2d = []

        for feature in results:
            if hasattr(results[feature][1], "__iter__"):
                if len(results[feature][1].shape) == 0:
                    features_0d.append(feature)
                elif len(results[feature][1].shape) == 1:
                    features_1d.append(feature)
                else:
                    features_2d.append(feature)
            else:
                features_0d.append(feature)

        return features_0d, features_1d, features_2d
