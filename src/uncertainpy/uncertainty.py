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
                 save_figures=True,
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
           plot_condensed=True,
           plot_simulator_results=False,
           output_dir_figures=None,
           output_dir_data=None,
           filename=None,
           **custom_kwargs):
        """
method: pc, mc
pc_method: "regression"
        """
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)


        if method.lower() == "pc":
            if single:
                self.PCSingle(uncertain_parameters=uncertain_parameters,
                              method=pc_method,
                              rosenblatt=rosenblatt,
                              plot_condensed=plot_condensed,
                              plot_simulator_results=plot_simulator_results,
                              output_dir_figures=output_dir_figures,
                              output_dir_data=output_dir_data,
                              filename=filename)
            else:
                self.PC(uncertain_parameters=uncertain_parameters,
                        method=pc_method,
                        rosenblatt=rosenblatt,
                        plot_condensed=plot_condensed,
                        plot_simulator_results=plot_simulator_results,
                        output_dir_figures=output_dir_figures,
                        output_dir_data=output_dir_data,
                        filename=filename)
        elif method.lower() == "mc":
            if single:
                self.MCSingle(uncertain_parameters=uncertain_parameters,
                              plot_condensed=plot_condensed,
                              plot_simulator_results=plot_simulator_results,
                              output_dir_figures=output_dir_figures,
                              output_dir_data=output_dir_data,
                              filename=filename)
            else:
                self.MC(uncertain_parameters=uncertain_parameters,
                        plot_condensed=plot_condensed,
                        plot_simulator_results=plot_simulator_results,
                        output_dir_figures=output_dir_figures,
                        output_dir_data=output_dir_data,
                        filename=filename)

        elif method.lower() == "custom":
            self.CustomUQ(uncertain_parameters=uncertain_parameters,
                          plot_condensed=plot_condensed,
                          plot_simulator_results=plot_simulator_results,
                          output_dir_figures=output_dir_figures,
                          output_dir_data=output_dir_data,
                          filename=filename,
                          **custom_kwargs)



    def CustomUQ(self,
                 plot_condensed=True,
                 plot_simulator_results=False,
                 output_dir_figures=None,
                 output_dir_data=None,
                 filename=None,
                 **custom_kwargs):

        self.data = self.uncertainty_calculations.CustomUQ(**custom_kwargs)

        if self.save_data:
            self.save(self.output_data_filename, output_dir=output_dir_data)

        if self.save_figures:
            self.plot(condensed=plot_condensed, output_dir=output_dir_figures)

        if plot_simulator_results:
            self.plot(simulator_results=True)


    def PC(self,
           uncertain_parameters=None,
           method="regression",
           rosenblatt=False,
           plot_condensed=True,
           plot_simulator_results=False,
           output_dir_figures=None,
           output_dir_data=None,
           filename=None):

        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high. A Monte-Carlo method _might_ be faster.")


        self.data = self.uncertainty_calculations.PC(
            uncertain_parameters=uncertain_parameters,
            method=method,
            rosenblatt=rosenblatt
        )


        if self.save_data:
            if filename is None:
                filename = self.output_data_filename

            self.save(filename, output_dir=output_dir_data)

        if self.save_figures:
            self.plot(condensed=plot_condensed, output_dir=output_dir_figures)

        if plot_simulator_results:
            self.plot(simulator_results=True, output_dir=output_dir_figures)



    def MC(self,
           uncertain_parameters=None,
           plot_condensed=True,
           plot_simulator_results=False,
           output_dir_figures=None,
           output_dir_data=None,
           filename=None):

        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        self.data = self.uncertainty_calculations.MC(uncertain_parameters=uncertain_parameters)

        if self.save_data:
            if filename is None:
                filename = self.output_data_filename

            self.save(filename, output_dir=output_dir_data)


        if self.save_figures:
            self.plot(condensed=plot_condensed,
                      sensitivity=False,
                      output_dir=output_dir_figures)

        if plot_simulator_results:
            self.plot(simulator_results=True, output_dir=output_dir_figures)





    def PCSingle(self,
                 uncertain_parameters=None,
                 method="regression",
                 rosenblatt=False,
                 plot_condensed=True,
                 plot_simulator_results=False,
                 output_dir_data=None,
                 output_dir_figures=None,
                 filename=None):

        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high. A Monte-Carlo method _might_ be faster.")


        if filename is None:
            filename = self.output_data_filename


        for uncertain_parameter in uncertain_parameters:
            self.logger.info("Running for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.PC(
                uncertain_parameters=uncertain_parameter,
                method=method,
                rosenblatt=rosenblatt
            )

            tmp_filename = "{}_single-parameter-{}".format(
                filename,
                uncertain_parameter
            )

            if self.save_data:
                self.save(tmp_filename, output_dir=output_dir_data)


            if output_dir_figures is None:
                tmp_output_dir_figures = os.path.join(self.output_dir_figures, tmp_filename)
            else:
                tmp_output_dir_figures = os.path.join(output_dir_figures, tmp_filename)


            if self.save_figures:
                self.plot(condensed=plot_condensed,
                          sensitivity=False,
                          output_dir=tmp_output_dir_figures)

            if plot_simulator_results:
                self.plot(simulator_results=True,
                          output_dir=tmp_output_dir_figures)


    def MCSingle(self,
                 uncertain_parameters=None,
                 plot_condensed=True,
                 plot_simulator_results=False,
                 output_dir_data=None,
                 output_dir_figures=None,
                 filename=None):
        uncertain_parameters = self.convertUncertainParameters(uncertain_parameters)

        if filename is None:
            filename = self.output_data_filename


        for uncertain_parameter in uncertain_parameters:
            self.logger.info("Running MC for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.MC(uncertain_parameter)

            tmp_filename = "{}_single-parameter-{}".format(
                filename,
                uncertain_parameter
            )

            if self.save_data:
                self.save(tmp_filename, output_dir=output_dir_data)


            if output_dir_figures is None:
                tmp_output_dir_figures = os.path.join(self.output_dir_figures, tmp_filename)
            else:
                tmp_output_dir_figures = os.path.join(output_dir_figures, tmp_filename)

            if self.save_figures:
                self.plot(condensed=plot_condensed, output_dir=tmp_output_dir_figures)

            if plot_simulator_results:
                self.plot(simulator_results=True, output_dir=tmp_output_dir_figures)



    def save(self, filename, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir_data

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_path = os.path.join(output_dir, filename + ".h5")

        self.logger.info("Saving data as: {}".format(save_path))

        ### TODO expand the save funcition to also save parameters and model information
        self.data.save(save_path)



    def load(self, filename):
        self.data = Data(filename)


    def plot(self,
             condensed=True,
             sensitivity=True,
             simulator_results=False,
             output_dir=None):

        if output_dir is None:
            output_dir = self.output_dir_figures

        self.plotting.setData(self.data, output_dir=output_dir)

        self.logger.info("Plotting in {}".format(output_dir))

        if simulator_results:
            self.plotting.plotSimulatorResults()
        else:
            self.plotting.plot(condensed=condensed,
                               sensitivity=sensitivity)



    def convertUncertainParameters(self, uncertain_parameters):
        if uncertain_parameters is None:
            uncertain_parameters = self.model.parameters.getUncertain("name")

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        return uncertain_parameters
