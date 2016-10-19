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

# TODO find out where to save parameter distributions and parameter objects. In
# the least remove the problems with parameters and setParameters beeing two
# different things

# TODO implement a feature description in features


# TODO Make sending of model keywords simpler than it currently is. Use a
# dictionary similar to kwargs?. model.py

# TODO Make 3d plots of 2d features

# TODO calculate sensitivity in MC method

# TODO combine save_data and save_data_dir into the save variable?


# TODO Profile the code using line profiler or
# python -m cProfile %1


# TODO in CoffeeCupPointModel set kappa=beta*phi and see how dependent variables work.

# TODO make it possible to create dependent variables.

# TODO Move plotDirectComparison to plotUncertainty.

# TODO Add support for 2d interpolation

# TODO Refactor so private variables are "hidden", starts with _

# TODO reduce the number of t values stored.
# When not an adaptive model only 1 set is needed to be stored

# TODO Does it make any sense to perform an interpolation for features?
# Currently features are excluded from interpolation

# TODO move creation of all_features to where the class is initiated

# TODO save data from each singleParameter run or not in the class?


# Figures are always saved on the format:
# output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension


import time
import os

import numpy as np
import chaospy as cp
import multiprocessing as mp
import logging
from tqdm import tqdm

from xvfbwrapper import Xvfb

# Imported from uncertainpy
from uncertainpy.features import NeuronFeatures
from uncertainpy.evaluateNodeFunction import evaluateNodeFunction
from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.utils import create_logger
from uncertainpy import Data

class UncertaintyEstimation():
    def __init__(self, model,
                 features=None,
                 save_figures=False,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 save_data=True,
                 output_dir_data="data/",
                 output_data_filename=None,
                 supress_model_graphics=True,
                 supress_model_output=True,
                 CPUs=mp.cpu_count(),
                 rosenblatt=False,
                 nr_mc_samples=10**3,
                 nr_pc_mc_samples=10**5,
                 verbose_level="info",
                 verbose_filename=None,
                 logger=None,
                 seed=None):
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


        self.data = Data()


        # TODO there is something weird with features here. the Default should
        # probably not be NeuronFeatures
        if features is None:
            self.features = NeuronFeatures()
        else:
            self.features = features

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

        self.M = 3
        self.nr_mc_samples = nr_mc_samples
        self.nr_pc_mc_samples = nr_pc_mc_samples


        self.resetValues()




        if logger is None:
            self.logger = create_logger(verbose_level,
                                        verbose_filename,
                                        self.__class__.__name__)
        else:
            self.logger = logger

        self.plot = PlotUncertainty(data_dir=self.output_dir_data,
                                    output_dir_figures=output_dir_figures,
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



    # def __del__(self):
    #     pass


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

        self.U_hat = {}
        self.distribution = None
        self.solves = None
        self.U_mc = {}  # TODO Is this needed to save?
        self.P = None

        self.data.resetValues()

        self.data.xlabel = self.model.xlabel
        self.data.ylabel = self.model.ylabel


    def evaluateNodeFunctionList(self, nodes):
        data = []

        for node in nodes:
            data.append((self.model.cmd(),
                         self.supress_model_output,
                         self.model.adaptive_model,
                         node,
                         self.data.uncertain_parameters,
                         self.features.cmd(),
                         self.features.kwargs()))
        return data



    def evaluateNodes(self, nodes):

        if self.supress_model_graphics:
            vdisplay = Xvfb()
            vdisplay.start()

        solves = []
        pool = mp.Pool(processes=self.CPUs)
        # solves = pool.map(evaluateNodeFunction,
        #                   self.evaluateNodeFunctionList(nodes.T))
        #
        for result in tqdm(pool.imap(evaluateNodeFunction,
                                     self.evaluateNodeFunctionList(nodes.T)),
                           desc="Running model",
                           total=len(nodes.T)):


            solves.append(result)


        pool.close()

        if self.supress_model_graphics:
            vdisplay.stop()

        return np.array(solves)



    def sortFeaturesFromResults(self, results):
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
        features_0d, features_1d, features_2d = self.sortFeaturesFromResults(solves[0])


        self.data.features_0d = features_0d
        self.data.features_1d = features_1d
        self.data.features_2d = features_2d
        self.data.feature_list = features_0d + features_1d + features_2d

        self.data.feature_list.sort()

        for feature in self.data.features_2d:
            if self.model.adaptive_model and feature == "directComparison":
                raise NotImplementedError("Support for >= 2d interpolation is not yet implemented")

            else:
                self.data.t[feature] = solves[0][feature][0]

                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature][1])

                # self.U[feature] = np.array(self.U[feature])

        for feature in self.data.features_1d:
            if self.model.adaptive_model and feature == "directComparison":
                ts = []
                interpolation = []
                for solved in solves:
                    ts.append(solved[feature][0])
                    interpolation.append(solved[feature][2])

                self.data.t[feature], self.data.U[feature] = self.performInterpolation(ts, interpolation)
            else:
                self.data.t[feature] = solves[0][feature][0]
                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature][1])

                # self.data.U[feature] = np.array(self.U[feature])


        for feature in self.data.features_0d:
            self.data.U[feature] = []
            self.data.t[feature] = None
            for solved in solves:
                self.data.U[feature].append(solved[feature][1])

            # self.U[feature] = np.array(self.U[feature])

        # self.t[feature] = np.array(self.t[feature])
        self.data.U[feature] = np.array(self.data.U[feature])




    def createMask(self, nodes, feature):

        if feature not in self.data.feature_list:
            raise AttributeError("Error: {} is not a feature".format(feature))

        i = 0
        masked_U = []
        masked_nodes = []
        mask = np.ones(len(self.data.U[feature]), dtype=bool)

        for result in self.data.U[feature]:
            if not isinstance(result, np.ndarray) and np.isnan(result):
                mask[i] = False
            else:
                if feature in self.data.feature_list:
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
            self.data.uncertain_parameters = self.model.parameters.getUncertain()
        else:
            parameter_space = [self.model.parameters.get(parameter_name).parameter_space]
            self.data.uncertain_parameters = [parameter_name]



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

        # Running the model
        solves = self.evaluateNodes(nodes)

        # Store the results from the runs in self.U and self.t, and interpolate U if there is a t
        self.storeResults(solves)

        # TODO keep this test for adaptive_model ?
        # test if it is an adaptive model
        if not self.model.adaptive_model:
            for feature in self.data.features_1d + self.data.features_2d:
                u_prev = self.data.U[feature][0]
                for u in self.data.U[feature][1:]:
                    if u_prev.shape != u.shape:
                        raise ValueError("The number of simulation points varies between simulations. Try setting adaptive_model=True in model()")
                    u_prev = u


        # Calculate PC for each feature
        for feature in self.data.feature_list:
            if self.rosenblatt:
                masked_nodes, masked_U = self.createMask(nodes_MvNormal, feature)

            else:
                masked_nodes, masked_U = self.createMask(nodes, feature)

            # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
            self.U_hat[feature] = cp.fit_regression(self.P, masked_nodes,
                                                    masked_U, rule="T")

        # # perform for directComparison outside, since masking is not needed.
        # # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
        # self.U_hat["directComparison"] = cp.fit_regression(self.P, nodes,
        #                                                    self.data.U["directComparison"], rule="T")


    def PCAnalysis(self):
        for feature in self.data.feature_list:
            self.data.E[feature] = cp.E(self.U_hat[feature], self.distribution)
            self.data.Var[feature] = cp.Var(self.U_hat[feature], self.distribution)

            samples = self.distribution.sample(self.nr_pc_mc_samples, "R")

            if len(self.data.uncertain_parameters) > 1:

                self.U_mc[feature] = self.U_hat[feature](*samples)
                self.data.sensitivity_1[feature] = cp.Sens_m(self.U_hat[feature], self.distribution)
                self.data.sensitivity_t[feature] = cp.Sens_t(self.U_hat[feature], self.distribution)

            else:
                self.U_mc[feature] = self.U_hat[feature](samples)
                self.data.sensitivity_1[feature] = None
                self.data.sensitivity_t[feature] = None


            self.data.p_05[feature] = np.percentile(self.U_mc[feature], 5, -1)
            self.data.p_95[feature] = np.percentile(self.U_mc[feature], 95, -1)

        self.totalSensitivity(sensitivity="sensitivity_1")
        self.totalSensitivity(sensitivity="sensitivity_t")



    def MC(self, parameter_name=None):
        if parameter_name is None:
            parameter_space = self.model.parameters.getUncertain("parameter_space")
            self.data.uncertain_parameters = self.model.parameters.getUncertain()
        else:
            parameter_space = [self.model.parameters.get(parameter_name).parameter_space]
            self.data.uncertain_parameters = [parameter_name]

        self.distribution = cp.J(*parameter_space)
        nodes = self.distribution.sample(self.nr_mc_samples, "M")

        solves = self.evaluateNodes(nodes)

        # Find 1d and 2d features
        # Store the results from the runs in self.U and self.t, and interpolate U if there is a t
        self.storeResults(solves)

        for feature in self.data.feature_list:
            self.data.E[feature] = np.mean(self.data.U[feature], 0)
            self.data.Var[feature] = np.var(self.data.U[feature], 0)

            self.data.p_05[feature] = np.percentile(self.data.U[feature], 5, 0)
            self.data.p_95[feature] = np.percentile(self.data.U[feature], 95, 0)
            self.data.sensitivity_1[feature] = None
            self.data.total_sensitivity_1[feature] = None
            self.data.sensitivity_t[feature] = None
            self.data.total_sensitivity_t[feature] = None


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

                self.save(filename)

            if self.save_figures:
                filename = "%s_single-parameter-%s" \
                    % (self.output_data_filename, uncertain_parameter)
                self.logger.info("Saving plots as: {}".format(filename))
                self.plotAllSingle(filename)



    def allParameters(self):
        self.resetValues()

        self.logger.info("Running for all parameters")
        if self.createPCExpansion() == -1:
            self.logger.warning("Calculations aborted for all")
            return -1

        self.PCAnalysis()

        if self.save_data:
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
                self.save(filename)


            if self.save_figures:
                self.logger.info("Saving plots as: {}".format(filename))
                self.plotAllSingle(filename)


    def allParametersMC(self):
        # if len(self.model.parameters.uncertain_parameters) <= 1:
        #     print "Only 1 uncertain parameter"
        #     return

        self.resetValues()

        self.MC()

        if self.save_data:
            self.save(self.output_data_filename)


        if self.save_figures:
            self.logger.info("Saving plots as: {}".format(self.output_data_filename))
            self.plotAllSingle(self.output_data_filename)


    def getData(self):
        return self.data



    def totalSensitivity(self, sensitivity="sensitivity_1"):

        sense = getattr(self.data, sensitivity)
        total_sense = {}

        for feature in sense:
            if sense[feature] is None:
                continue

            total_sensitivity = 0
            total_sense[feature] = []
            for i in xrange(0, len(self.data.uncertain_parameters)):
                tmp_sum_sensitivity = np.sum(sense[feature][i])

                total_sensitivity += tmp_sum_sensitivity
                total_sense[feature].append(tmp_sum_sensitivity)

            for i in xrange(0, len(self.data.uncertain_parameters)):
                if not total_sensitivity == 0:
                    total_sense[feature][i] /= float(total_sensitivity)


        setattr(self.data, "total_" + sensitivity, total_sense)


    def save(self, filename):
        if not os.path.isdir(self.output_dir_data):
            os.makedirs(self.output_dir_data)

        self.logger.info("Saving data as: {}".format(filename))

        ### TODO expand the save funcition to also save parameters and model information
        self.data.save(os.path.join(self.output_dir_data, filename + ".h5"))


    # TODO never tested
    def load(self, filename):
        self.filename = filename
        self.data.load(os.path.join(self.data_dir, filename + ".h5"))


    def plotAll(self, foldername=None):
        self.plot.setData(self.data, foldername=foldername)

        if foldername is None:
            foldername = self.output_dir_figures


        self.plot.plotAllDataSensitivity()


    def plotAllSingle(self, foldername=None):
        self.plot.setData(self.data, foldername=foldername)

        if foldername is None:
            foldername = self.output_dir_figures


        self.plot.plotAllDataNoSensitivity()



    def plotResults(self, foldername=None):
        self.plot.setData(self.data, foldername=foldername)

        self.plot.plotResults()
