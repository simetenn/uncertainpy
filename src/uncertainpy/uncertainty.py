#  Figures are always saved on the format:
#  output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension

import os



# Imported from uncertainpy
from uncertainty_calculations import UncertaintyCalculations
from uncertainpy.features import GeneralFeatures
from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.utils import create_logger



class UncertaintyEstimation():
    def __init__(self, model,
                 features=None,
                 uncertainty_calculations=None,
                 save_figures=False,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 save_data=True,
                 output_dir_data="data/",
                 output_data_filename=None,
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
    Default is True.
CPUs : int
    The number of CPUs to perform
    calculations on.
    Default is mp.cpu_count() - the number of CPUs on your computer
rosenblatt : False
    If a rosenblatt transformation should be used.
    Use this if you have dependent uncertain
    parameters.
    Note: not been tested on dependent uncertain
    parameters.
    Default is False.
nr_mc_samples : int
    The number of samples usend when performing a Monte Carlo
    Method.
    Default is 10**3.
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
seed : None | int | array_like, optional
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

        self.data = None

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        self.save_figures = save_figures
        self.save_data = save_data
        self.output_dir_data = output_dir_data
        self.output_dir_figures = output_dir_figures

        self.model = model

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        if uncertainty_calculations is None:
            self.uncertainty_calculations = UncertaintyCalculations(
                model,
                features=self.features,
                rosenblatt=rosenblatt,
                verbose_level=verbose_level,
                verbose_filename=verbose_filename
            )
        else:
            self.uncertainty_calculations = uncertainty_calculations


        self.plot = PlotUncertainty(data_dir=self.output_dir_data,
                                    output_dir_figures=self.output_dir_figures,
                                    figureformat=figureformat,
                                    verbose_level=verbose_level,
                                    verbose_filename=verbose_filename)


        if output_data_filename is None:
            self.output_data_filename = self.model.__class__.__name__
        else:
            self.output_data_filename = output_data_filename


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


    def uq(self,
           uncertain_parameters=None,
           method="pc",
           single=False,
           pc_method="regression",
           rosenblatt=False):

        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if method == "pc":
            if single:
                self.PCSingle(uncertain_parameters=uncertain_parameters,
                              method=method,
                              rosenblatt=rosenblatt)
            else:
                self.PC(uncertain_parameters=uncertain_parameters,
                        method=method,
                        rosenblatt=rosenblatt)
        elif method == "mc":
            if single:
                self.MCSingle(uncertain_parameters=uncertain_parameters)
            else:
                self.MC(uncertain_parameters=uncertain_parameters)


    def PC(self, uncertain_parameters=None, method="regression", rosenblatt=False):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high. A Monte-Carlo method _might_ be faster.")

        self.data = self.uncertainty_calculations.PC(
            uncertain_parameters=uncertain_parameters,
            method=method,
            rosenblatt=rosenblatt
        )

        if self.save_data:
            self.save(self.output_data_filename)


        if self.save_figures:
            self.plotAll(self.output_data_filename)


    def MC(self, uncertain_parameters=None):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        self.data = self.uncertainty_calculations.MC()

        if self.save_data:
            self.save(self.output_data_filename)


        if self.save_figures:
            self.plotAll(self.output_data_filename)



    def PCSingle(self, uncertain_parameters=None, method="regression", rosenblatt=False):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high. A Monte-Carlo method _might_ be faster.")

        for uncertain_parameter in self.model.parameters.getUncertain():
            self.logger.info("Running for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.PC(
                uncertain_parameters=uncertain_parameters,
                method=method,
                rosenblatt=rosenblatt
            )


            if self.save_data:
                filename = "{}_single-parameter-{}".format(
                    self.output_data_filename,
                    uncertain_parameter
                )

                self.save(filename)

            if self.save_figures:
                filename = "{}_single-parameter-{}".format(
                    self.output_data_filename,
                    uncertain_parameter
                )

                self.plotAllSingle(filename)



    def MCSingle(self, uncertain_parameters=None):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        for uncertain_parameter in self.model.parameters.getUncertain():
            self.logger.info("Running MC for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.MC()

            if self.save_data:
                self.save(self.output_data_filename)

            if self.save_figures:
                self.plotAll(self.output_data_filename)




    def save(self, filename):
        if not os.path.isdir(self.output_dir_data):
            os.makedirs(self.output_dir_data)

        self.logger.info("Saving data as: {}".format(filename))

        ### TODO expand the save funcition to also save parameters and model information
        self.data.save(os.path.join(self.output_dir_data, filename + ".h5"))


    # TODO never tested
    def load(self, filename):
        raise NotImplementedError

        self.filename = filename
        self.data.load(os.path.join(self.data_dir, filename + ".h5"))


    def plotAll(self, foldername=None):
        self.logger.info("Creating plots as: {}".format(self.output_data_filename))

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


    def convertUncertainParameters(self, uncertain_parameters):
        if uncertain_parameters is None:
            uncertain_parameters = self.model.parameters.getUncertain("name")

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        return uncertain_parameters
