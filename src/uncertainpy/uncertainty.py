#  Figures are always saved on the format:
#  output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension


import time
import os

import numpy as np
import chaospy as cp
import multiprocessing as mp
import logging


# Imported from uncertainpy
from uncertainpy.features import GeneralFeatures
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
        """


        self.data = Data()


        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
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




        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)


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
        # self.seed = seed

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
