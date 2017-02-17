#  Figures are always saved on the format:
#  output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension

import os



# Imported from uncertainpy
from uncertainty_calculations import UncertaintyCalculations
from uncertainpy.features import GeneralFeatures
from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.utils import create_logger
from uncertainpy import Data


class UncertaintyEstimation():
    def __init__(self, model,
                 features=None,
                 uncertainty_calculations=None,
                 save_figures=False,
                 plot_type="results",
                 plot_simulator_results=False,
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
        self.plot_simulator_results = plot_simulator_results

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features


        self.model = model

        if uncertainty_calculations is None:
            self.uncertainty_calculations = UncertaintyCalculations(
                model=self.model,
                features=self.features,
                verbose_level=verbose_level,
                verbose_filename=verbose_filename
            )
        else:
            self.uncertainty_calculations = uncertainty_calculations
            self.uncertainty_calculations.set_model(model)
            self.uncertainty_calculations.set_features(self.features)


        self.plot_type = plot_type

        self.plotting = PlotUncertainty(output_dir=self.output_dir_figures,
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


    def UQ(self,
           uncertain_parameters=None,
           method="pc",
           single=False,
           pc_method="regression",
           rosenblatt=False,
           plot_type=None,
           **custom_kwargs):
        """
method: pc, mc

        """
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if plot_type is not None:
            self.plot_type = plot_type

        if method.lower() == "pc":
            if single:
                self.PCSingle(uncertain_parameters=uncertain_parameters,
                              method=pc_method,
                              rosenblatt=rosenblatt)
            else:
                self.PC(uncertain_parameters=uncertain_parameters,
                        method=pc_method,
                        rosenblatt=rosenblatt)
        elif method.lower() == "mc":
            if single:
                self.MCSingle(uncertain_parameters=uncertain_parameters)
            else:
                self.MC(uncertain_parameters=uncertain_parameters)

        elif method.lower() == "custom":
            self.CustomUQ(**custom_kwargs)



    def CustomUQ(self, **custom_kwargs):

        self.data = self.uncertainty_calculations.CustomUQ(**custom_kwargs)

        if self.save_data:
            self.save(self.output_data_filename)

        if self.save_figures:
            self.plot(plot_type=self.plot_type)




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
            self.plot(plot_type=self.plot_type)

        if self.plot_simulator_results:
            self.plot(plot_type="simulator_results")



    def MC(self, uncertain_parameters=None):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        self.data = self.uncertainty_calculations.MC(uncertain_parameters=uncertain_parameters)

        if self.save_data:
            self.save(self.output_data_filename)


        if self.save_figures:
            self.plot(plot_type=self.plot_type)

        if self.plot_simulator_results:
            self.plot(plot_type="simulator_results")





    def PCSingle(self, uncertain_parameters=None, method="regression", rosenblatt=False):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high. A Monte-Carlo method _might_ be faster.")

        for uncertain_parameter in uncertain_parameters:
            self.logger.info("Running for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.PC(
                uncertain_parameters=uncertain_parameter,
                method=method,
                rosenblatt=rosenblatt
            )

            filename = "{}_single-parameter-{}".format(
                self.output_data_filename,
                uncertain_parameter
            )

            if self.save_data:
                self.save(filename)

            if self.save_figures:
                self.plot(output_dir_figures=os.path.join(self.output_dir_figures,
                                                          filename),
                          plot_type=self.plot_type)

            if self.plot_simulator_results:
                self.plot(output_dir_figures=os.path.join(self.output_dir_figures,
                                                          filename),
                          plot_type="simulator_results")


    def MCSingle(self, uncertain_parameters=None):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        for uncertain_parameter in uncertain_parameters:
            self.logger.info("Running MC for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.MC(uncertain_parameter)

            filename = "{}_single-parameter-{}".format(
                self.output_data_filename,
                uncertain_parameter
            )

            if self.save_data:
                self.save(filename)

            if self.save_figures:
                self.plot(output_dir_figures=filename, plot_type=self.plot_type)

            if self.plot_simulator_results:
                self.plot(output_dir_figures=os.path.join(self.output_dir_figures,
                                                          filename),
                          plot_type="simulator_results")



    def save(self, filename):
        if not os.path.isdir(self.output_dir_data):
            os.makedirs(self.output_dir_data)

        self.logger.info("Saving data as: {}".format(filename))

        ### TODO expand the save funcition to also save parameters and model information
        self.data.save(os.path.join(self.output_dir_data, filename + ".h5"))



    def load(self, filename):
        self.data = Data(os.path.join(filename))


    def plot(self, plot_type="results", output_dir_figures=None):
        """
results
no_sensitivity
all
simulator_results
        """


        self.plotting.setData(self.data, output_dir=output_dir_figures)

        if plot_type == "results":
            self.plotting.plotResults()
        elif plot_type == "no_sensitivity":
            self.plotting.plotAllDataNoSensitivity()
        elif plot_type == "all":
            self.plotting.plotAllDataAllSensitivity()
        elif plot_type == "simulator_results":
            self.plotting.plotSimulatorResults()
        else:
            raise ValueError("Invalid plot type: {plot_type}".format(plot_type=plot_type))



    def convertUncertainParameters(self, uncertain_parameters):
        if uncertain_parameters is None:
            uncertain_parameters = self.model.parameters.getUncertain("name")

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        return uncertain_parameters
