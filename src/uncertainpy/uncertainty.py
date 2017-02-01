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
                 verbose_filename=None):

        self.data = None

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        self.save_figures = save_figures
        self.save_data = save_data
        self.output_dir_data = output_dir_data
        self.output_dir_figures = output_dir_figures

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        if uncertainty_calculations is None:
            self.uncertainty_calculations = UncertaintyCalculations(
                model=model,
                features=self.features,
                verbose_level=verbose_level,
                verbose_filename=verbose_filename
            )
        else:
            self.uncertainty_calculations = uncertainty_calculations
            self.uncertainty_calculations.set_model(model)
            self.model = model


        self.plot = PlotUncertainty(data_dir=self.output_dir_data,
                                    output_dir_figures=self.output_dir_figures,
                                    figureformat=figureformat,
                                    verbose_level=verbose_level,
                                    verbose_filename=verbose_filename)


        if output_data_filename is None:
            self.output_data_filename = self.model.__class__.__name__
        else:
            self.output_data_filename = output_data_filename




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
                self.MCSingle(uncertain_parameters=uncertain_parameters)


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
