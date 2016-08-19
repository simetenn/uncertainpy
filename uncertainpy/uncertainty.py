# TODO Test out different types of polynomial chaos methods

# TODO Test correlation

# TODO find a better way to store tmp files

# TODO, only use evaluateNodeFunction if it is a neuron model,
# withouth a neuronmodel we will not have problems with neuron and chaospy

# TODO Do a regression analysis to figure out which variables are dependent on
# each other

# TODO Use a recursive neural network to do something?

# TODO have the option of saving the exploration by parameters instead of by distribution


# TODO use sumatra when starting runs

# TODO save the entire class to file

# TODO Add support for aborting a simulation when it is above a certain memory theeshold

# TODO incorporate singleNeuronRun into a class of file

# TODO problems with new neuron models, and their path

# TODO make it so setup.py install all requirements

# TODO make it so rosenblatt transform code looks pretier and easier

# TODO Rewrite model.py so it use kwargs to send options back and forth instead of argparse

# TODO find out where to save parameter distributions and parameter objects. In
# the least remove the problems with parameters and setParameters beeing two
# different things

# TODO implement a feature description in features

# TODO Reimplement sensitivity ranking, but for all features and not only the diirect_comparison

# TODO placement of feature bar plots

# TODO Decide if the HH model should return Vm - 65 or just Vm

# TODO Make sending of model keywords simpler than it currently is. Use a
# dictionary similar to kwargs?

# TODO create a progressbar using tqdm


# TODO Make 3d plots of 2d features

# TODO calculate sensitivity in MC method

# TODO rewrite plotting code to be a series of functions?

# TODO combine save_data and save_data_dir into the save variable?

# TODO use _ before hidden variables

# TODO Add a option for models with variable timesteps

# TODO Profile the code using line profiler or
# python -m cProfile %1

### After meeting

# TODO Variance of mean and variance of the model is two seperate things.
# Variance of mean tells how well I have calculated the mean while
# variance of the model tells how well the model is.
# Make sure that I calculate the correct one of those in PC and MC method
# Also make sure I compare the same ones when comparing MC with PC

# TODO in CoffeeCupPointModel set kappa=beta*phi and see how dependent variables work.

# TODO make it possible to create dependent variables.

# TODO Move plotDirectComparison to plotUncertainty.

# TODO Create full test coverage

# TODO Add support for 2d interpolation

# TODO Refactor so private variables are "hidden", starts with _

# TODO reduce the number of t values stored.
# When not an adaptive model only 1 set is needed to be stored

# TODO Does it make any sense to perform an interpolation for features?
# Currently features are excluded from interpolation

# TODO finish testing

# TODO move creation of all_features to where the class is initiated

# TODO save data from each singleParameter run or not in the class?

# TODO remove feature list from UncertaintyEstimation, and add it to feature class instead

# TODO make it possible to use ones own plotting class


# Figures are always saved on the format:
# output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension


import time
import os
import h5py
import sys

import numpy as np
import chaospy as cp
import multiprocessing as mp
import matplotlib.pyplot as plt
import logging


from xvfbwrapper import Xvfb

# Imported from uncertainpy
from uncertainpy.features import NeuronFeatures
from uncertainpy.evaluateNodeFunction import evaluateNodeFunction
from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.plotting.prettyPlot import prettyPlot
from uncertainpy.utils import create_logger


class UncertaintyEstimation():
    def __init__(self, model,
                 features=None,
                 feature_list=None,
                 save_figures=False,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 combined_features=True,
                 save_data=True,
                 output_dir_data="data/",
                 output_dir_gif="gif/",
                 output_data_filename=None,
                 supress_model_graphics=True,
                 supress_model_output=True,
                 CPUs=mp.cpu_count(),
                 rosenblatt=False,
                 nr_mc_samples=10**3,
                 nr_pc_mc_samples=10**5,
                 verbose_level="info",
                 verbose_filename=None,
                 seed=None,
                 feature_options={}):
        """
Uncertainty Estimation object

Parameters
----------
Required arguments

model : Model Object
    The model on which to quantify uncertaintes.

Optional arguments

features : Feature Object
    Default is NeuronFeatures
feature_list : list
    A list of all features to be calculated.
    Default is None, where no features will be calculated.
save_figures : bool
    If figures should be created and saved.
    This is deprecated code from before many changes
    and might therefore not work. Recommend using
    PlotUncertainty and plotting from saved files
    instead.
    Default is False
output_dir_figures : str
    Folder where figures is saved.
    Default is figures/.
figureformat : ".png"
    The format figures are saved in.
    Matplotlib is used to plot, and most matplotlib backends support
    png, pdf, ps, eps and svg.
    Default is .png
save_data : bool
    If data should be saved to a hdf5 file.
    Default is True.
output_dir_data : str
    Folder where the data is saved.
    Default is "data/".
output_data_filename : str
    The name of the output file.
    Default is the name of the model.
supress_model_graphics : bool
    If the graphical output from the model
    should be supressed, such as GUI.
    Escpecially usefull for the Neuron simulator models.
    Default is True.
supress_model_output : True
    Supress terminal output from the model.
    Note: set this to false when debugging your model,
    otherwise print statements will be supressed.
    Defualt is True.
CPUs : int
    The number of CPUs to perform
    calculations on.
    Defualt is mp.cpu_count() - the number of CPUs on your computer
rosenblatt : False
    If a rosenblatt transformation should be used.
    Use this if you have dependent uncertain
    parameters.
    Note: not been tested on dependent uncertain
    parameters.
    Defualt is False.
nr_mc_samples : int
    The number of samples usend when performing a Monte Carlo
    Method.
    Defualt is 10**3.
nr_pc_mc_samples : int
    The number of samples when using the polynomal chaos
    polynomial as a surrogate model for a Monte Carlo method.
    Default is 10**5.
verbose_level : str
    The amount of information printed to the terminal.
    Options from much output to low output:
    'debug', 'info', 'warning', 'error','critical'
    Default is 'warnings'.
verbose_filename : None/str
    redirect output to a file. If None print to terminal.
    Default is None.
seed : None/ int or array_like, optional
    Setting a seed, usefull for testing purposes
feature_options : dict
    Optional arguments to be sent to features.

Methods
-------

resetValues
evaluateNodeFunctionList
createPCExpansion
evaluateNodes
singleParameterPCAnalysis
PCAnalysis
MC
timePassed
singleParameters
allParameters
singleParametersMC
allParametersMC
sensitivityRanking
save

Returns
-------

Examples
--------
For example on use see:
    hodgkin-huxley_uncertainty_estimation.py
    izhikevich_uncertainty_estimation.py
    lgn_uncertainty_estimation.py
    coffe_uncertainty_estimation.py
        """


        # TODO there is something weird with features here. the Default should
        # probably not be NeuronFeatures
        if features is None:
            self.features = NeuronFeatures()
        else:
            self.features = features

        if feature_list == "all":
            self.feature_list = self.features.implementedFeatures()
        elif feature_list is None:
            self.feature_list = []
        elif isinstance(feature_list, str):
            self.feature_list = [feature_list]
        else:
            self.feature_list = feature_list

        self.save_figures = save_figures
        self.figureformat = figureformat
        self.save_data = save_data
        self.output_dir_data = output_dir_data
        self.output_dir_figures = output_dir_figures

        self.supress_model_graphics = supress_model_graphics
        self.supress_model_output = supress_model_output
        self.CPUs = CPUs

        self.model = model

        self.rosenblatt = rosenblatt

        self.feature_options = feature_options
        # if self.warning_flag:
        #     warnings.simplefilter('ignore')

        self.M = 3
        self.nr_mc_samples = nr_mc_samples
        self.nr_pc_mc_samples = nr_pc_mc_samples

        self.resetValues()

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


        self.plot = PlotUncertainty(data_dir=self.output_dir_data,
                                    output_dir_figures=output_dir_figures,
                                    output_dir_gif=output_dir_gif,
                                    figureformat=figureformat,
                                    verbose_level=verbose_level,
                                    verbose_filename=verbose_filename)


        if output_data_filename is None:
            self.output_data_filename = self.model.__class__.__name__
        else:
            self.output_data_filename = output_data_filename



        if seed is not None:
            cp.seed(seed)
            np.random.seed(seed)
        self.seed = seed

        self.t_start = time.time()



    def __del__(self):
        pass


    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['pool']
    #     return self_dict
    #
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)


    def resetValues(self):
        self.parameter_names = None
        self.parameter_space = None

        self.U = {}
        self.U_hat = {}
        self.distribution = None
        self.solves = None
        self.t = {}
        self.E = {}
        self.Var = {}
        self.U_mc = {}  # TODO Is this needed to save?
        self.p_05 = {}
        self.p_95 = {}
        self.sensitivity = {}
        self.P = None



    def evaluateNodeFunctionList(self, nodes):
        data = []

        for node in nodes:
            data.append((self.model.cmd(),
                         self.supress_model_output,
                         self.model.adaptive_model,
                         node,
                         self.uncertain_parameters,
                         self.feature_list,
                         self.features.cmd(),
                         self.feature_options))
        return data



    def evaluateNodes(self, nodes):
        if self.supress_model_graphics:
            vdisplay = Xvfb()
            vdisplay.start()

        solves = []
        try:
            if self.CPUs > 1:
                pool = mp.Pool(processes=self.CPUs)
                solves = pool.map(evaluateNodeFunction,
                                  self.evaluateNodeFunctionList(nodes.T))
                pool.close()
            else:
                for node in nodes.T:
                    solves.append(evaluateNodeFunction([self.model.cmd(),
                                                        self.supress_model_output,
                                                        self.model.adaptive_model,
                                                        node,
                                                        self.uncertain_parameters,
                                                        self.feature_list,
                                                        self.features.cmd(),
                                                        self.feature_options]))
        except MemoryError:
            return -1

        if self.supress_model_graphics:
            vdisplay.stop()

        return np.array(solves)



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



    def performInterpolation(self, ts, interpolation):
        lengths = []
        for s in ts:
            lengths.append(len(s))

        index_max_len = np.argmax(lengths)
        t = ts[index_max_len]

        interpolated_solves = []
        for inter in interpolation:
            interpolated_solves.append(inter(t))

        interpolated_solves = np.array(interpolated_solves)

        return t, interpolated_solves



    def storeResults(self, solves):
        self.features_0d, self.features_1d, self.features_2d = self.sortFeatures(solves[0])

        self.all_features = self.features_0d + self.features_1d + self.features_2d

        for feature in self.features_2d:
            if self.model.adaptive_model and feature == "directComparison":
                raise NotImplementedError("Support for >= 2d interpolation is not yet implemented")

            else:
                self.t[feature] = solves[0][feature][0]

                self.U[feature] = []
                for solved in solves:
                    self.U[feature].append(solved[feature][1])

                # self.U[feature] = np.array(self.U[feature])

        for feature in self.features_1d:
            if self.model.adaptive_model and feature == "directComparison":
                ts = []
                interpolation = []
                for solved in solves:
                    ts.append(solved[feature][0])
                    interpolation.append(solved[feature][2])

                self.t[feature], self.U[feature] = self.performInterpolation(ts, interpolation)
            else:
                self.t[feature] = solves[0][feature][0]
                self.U[feature] = []
                for solved in solves:
                    self.U[feature].append(solved[feature][1])

                # self.U[feature] = np.array(self.U[feature])


        for feature in self.features_0d:
            self.U[feature] = []
            self.t[feature] = None
            for solved in solves:
                self.U[feature].append(solved[feature][1])

            # self.U[feature] = np.array(self.U[feature])

        # self.t[feature] = np.array(self.t[feature])
        self.U[feature] = np.array(self.U[feature])




    def createMask(self, nodes, feature):

        if feature not in self.all_features:
            raise AttributeError("Error: {} is not a feature".format(feature))

        i = 0
        masked_U = []
        masked_nodes = []
        mask = np.ones(len(self.U[feature]), dtype=bool)

        for result in self.U[feature]:
            if not isinstance(result, np.ndarray) and np.isnan(result):
                mask[i] = False
            else:
                if feature in self.all_features:
                    masked_U.append(result)

                else:
                    raise AttributeError("{} is not a feature".format(feature))

            i += 1


        if len(nodes.shape) > 1:
            masked_nodes = nodes[:, mask]
        else:
            masked_nodes = nodes[mask]


        if not np.all(mask):
            # raise RuntimeWarning("Feature: {} does not yield results for all parameter combinations".format(feature))
            self.logger.warning("Feature: {} does not yield results for all parameter combinations".format(feature))

        return np.array(masked_nodes), np.array(masked_U)


    # TODO move all of the uq calculations into it's own class
    def createPCExpansion(self, parameter_name=None, nr_pc_samples=None):

        self.nr_pc_samples = nr_pc_samples

        # TODO find a good way to solve the parameter_name poblem
        if parameter_name is None:
            parameter_space = self.model.parameters.getUncertain("parameter_space")
            self.uncertain_parameters = self.model.parameters.getUncertain()
        else:
            parameter_space = [self.model.parameters.get(parameter_name).parameter_space]
            self.uncertain_parameters = [parameter_name]



        if self.rosenblatt:
            # Create the Multivariat normal distribution
            dist_MvNormal = []
            for parameter in self.model.parameters.getUncertain("value"):
                dist_MvNormal.append(cp.Normal())

            dist_MvNormal = cp.J(*dist_MvNormal)


            self.distribution = cp.J(*parameter_space)
            self.P = cp.orth_ttr(self.M, dist_MvNormal)

            if self.nr_pc_samples is None:
                self.nr_pc_samples = 2*len(self.P) + 1

            nodes_MvNormal = dist_MvNormal.sample(self.nr_pc_samples, "M")
            # nodes_MvNormal, weights_MvNormal = cp.generate_quadrature(3, dist_MvNormal,
            #                                                           rule="J", sparse=True)

            nodes = self.distribution.inv(dist_MvNormal.fwd(nodes_MvNormal))
            # weights = weights_MvNormal*\
            #    self.distribution.pdf(nodes)/dist_MvNormal.pdf(nodes_MvNormal)

            self.distribution = dist_MvNormal

        else:
            self.distribution = cp.J(*parameter_space)

            self.P = cp.orth_ttr(self.M, self.distribution)

            if self.nr_pc_samples is None:
                self.nr_pc_samples = 2*len(self.P) + 2


            nodes = self.distribution.sample(self.nr_pc_samples, "M")
            # nodes, weights = cp.generate_quadrature(3, self.distribution, rule="J", sparse=True)

        # if len(nodes.shape) == 1:
        #     nodes = np.array([nodes])


        solves = self.evaluateNodes(nodes)

        # Store the results from the runs in self.U and self.t, and interpolate U if there is a t
        print "---------solves----------"
        print solves[0].keys()
        for i in solves:
            print len(i["directComparison"][0])
            print len(i["directComparison"][1])

        self.storeResults(solves)

        # Calculate PC for each feature
        for feature in self.all_features:
            if self.rosenblatt:
                masked_nodes, masked_U = self.createMask(nodes_MvNormal, feature)

            else:
                masked_nodes, masked_U = self.createMask(nodes, feature)

            # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)


            print "---------p----------"
            print self.P
            print "--------masked_nodes-----------"
            print masked_nodes.shape
            print "--------masked_U-----------"
            print masked_U.shape

            self.U_hat[feature] = cp.fit_regression(self.P, masked_nodes,
                                                    masked_U, rule="T")



    def PCAnalysis(self):
        for feature in self.all_features:
            self.E[feature] = cp.E(self.U_hat[feature], self.distribution)
            self.Var[feature] = cp.Var(self.U_hat[feature], self.distribution)

            if len(self.uncertain_parameters) > 1:
                    self.sensitivity[feature] = cp.Sens_t(self.U_hat[feature], self.distribution)

            samples = self.distribution.sample(self.nr_pc_mc_samples, "R")


            if len(self.uncertain_parameters) > 1:
                self.U_mc[feature] = self.U_hat[feature](*samples)
            else:
                self.U_mc[feature] = self.U_hat[feature](samples)

            self.p_05[feature] = np.percentile(self.U_mc[feature], 5, -1)
            self.p_95[feature] = np.percentile(self.U_mc[feature], 95, -1)



    def MC(self, parameter_name=None):
        if parameter_name is None:
            parameter_space = self.model.parameters.getUncertain("parameter_space")
            self.uncertain_parameters = self.model.parameters.getUncertain()
        else:
            parameter_space = [self.model.parameters.get(parameter_name).parameter_space]
            self.uncertain_parameters = [parameter_name]

        self.distribution = cp.J(*parameter_space)
        nodes = self.distribution.sample(self.nr_mc_samples, "M")

        solves = self.evaluateNodes(nodes)

        # Find 1d and 2d features
        # Store the results from the runs in self.U and self.t, and interpolate U if there is a t
        self.storeResults(solves)

        for feature in self.all_features:
            self.E[feature] = np.mean(self.U[feature], 0)
            self.Var[feature] = np.var(self.U[feature], 0)

            self.p_05[feature] = np.percentile(self.U[feature], 5, 0)
            self.p_95[feature] = np.percentile(self.U[feature], 95, 0)



    def timePassed(self):
        return time.time() - self.t_start


    def singleParameters(self):
        for uncertain_parameter in self.model.parameters.getUncertain():
            message = "Running for " + uncertain_parameter + "                     "
            self.logger.info(message)

            self.resetValues()

            if self.createPCExpansion(uncertain_parameter) == -1:
                self.logger.warning("Calculations aborted for " + uncertain_parameter)
                return -1

            self.PCAnalysis()


            if self.save_data:
                filename = "%s_single-parameter-%s" \
                    % (self.output_data_filename, uncertain_parameter)

                self.logger.info("Saving data as: {}".format(filename))
                self.save(filename)

            if self.save_figures:
                filename = "%s_single-parameter-%s" \
                    % (self.output_data_filename, uncertain_parameter)
                self.logger.info("Saving plots as: {}".format(filename))
                self.plotAll(filename)



    def allParameters(self):
        # if len(self.model.parameters.uncertain_parameters) <= 1:
        #     print "Only 1 uncertain parameter"
        #     return

        self.resetValues()

        self.logger.info("Running for all parameters")
        if self.createPCExpansion() == -1:
            self.logger.warning("Calculations aborted for all")
            return -1

        self.PCAnalysis()

        if self.save_data:
            self.logger.info("Saving data as: {}".format(self.output_data_filename))
            self.save(self.output_data_filename)


        if self.save_figures:
            self.logger.info("Saving plots as: {}".format(self.output_data_filename))
            self.plotAll(self.output_data_filename)



    def singleParametersMC(self):
        for uncertain_parameter in self.model.parameters.getUncertain():
            message = "Running MC for " + uncertain_parameter
            logging.info(message)

            self.resetValues()

            self.MC(uncertain_parameter)

            filename = "%s_single-parameter-%s" \
                % (self.output_data_filename, uncertain_parameter)

            if self.save_data:
                self.logger.info("Saving data as: {}".format(filename))
                self.save(filename)


            if self.save_figures:
                self.logger.info("Saving plots as: {}".format(filename))
                self.plotAll(filename)


    def allParametersMC(self):
        # if len(self.model.parameters.uncertain_parameters) <= 1:
        #     print "Only 1 uncertain parameter"
        #     return

        self.resetValues()

        self.MC()

        if self.save_data:
            self.logger.info("Saving data as: {}".format(self.output_data_filename))
            self.save(self.output_data_filename)


        if self.save_figures:
            self.logger.info("Saving plots as: {}".format(self.output_data_filename))
            self.plotAll(self.output_data_filename)


    # TODO does not work
    def sensitivityRanking(self):
        self.sensitivity_ranking = {}
        i = 0
        for parameter in self.model.parameters.getUncertain("name"):
            self.sensitivity_ranking[parameter] = sum(self.sensitivity["directComparison"][i])
            i += 1

        total_sensitivity = 0
        for parameter in self.sensitivity_ranking:
            total_sensitivity += self.sensitivity_ranking[parameter]

        for parameter in self.sensitivity_ranking:
            self.sensitivity_ranking[parameter] /= total_sensitivity


    def save(self, filename):
        if not os.path.isdir(self.output_dir_data):
            os.makedirs(self.output_dir_data)

        ### TODO expand the save funcition to also save parameters and model information

        f = h5py.File(os.path.join(self.output_dir_data, filename), 'w')

        # f.attrs["name"] = self.output_file.split("/")[-1]
        f.attrs["uncertain parameters"] = self.model.parameters.getUncertain("name")
        f.attrs["features"] = self.feature_list


        for feature in self.all_features:
            group = f.create_group(feature)

            if feature in self.t and self.t[feature] is not None:
                group.create_dataset("t", data=self.t[feature])
            # TODO do not save U to save space?
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
            if feature in self.sensitivity:
                group.create_dataset("sensitivity", data=self.sensitivity[feature])
            # if feature == "directComparison":
            #    group.create_dataset("total sensitivity", data=self.sensitivity_ranking[parameter])

        f.close()



    # TODO not finished
    def load(self, filename):
        self.filename = filename
        f = h5py.File(os.path.join(self.data_dir, self.filename), 'r')

        self.features_0d = []
        self.features_1d = []

        self.features_0d, self.features_1d = self.sortFeatures(self.E)


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
                self.t = f[feature]["t"][()]
            # else:
            #     self.t[feature] = None

        self.uncertain_parameters = f.attrs["uncertain parameters"]


    def plotSimulatorResults(self, foldername="simulator_results"):
        i = 1
        save_folder = os.path.join(self.output_dir_figures, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        padding = len(str(len(self.U["directComparison"]) + 1))
        for U in self.U["directComparison"]:
            prettyPlot(self.t["directComparison"], U,
                       xlabel="time", ylabel="voltage")
            plt.savefig(os.path.join(save_folder, "U_{0:0{1}d}".format(i, padding)))
            i += 1


    # TODO make sure this function works
    def plotAll(self):
        self.plot.setData(t=self.t,
                          U=self.U,
                          E=self.E,
                          Var=self.Var,
                          p_05=self.p_05,
                          p_95=self.p_95,
                          uncertain_parameters=self.uncertain_parameters,
                          sensitivity=self.sensitivity)


        self.plot.plotAllData()
