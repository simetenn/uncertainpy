#  Figures are always saved on the format:
#  output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension

import os
import multiprocess as mp
import types

from uncertainty_calculations import UncertaintyCalculations
from features import GeneralFeatures
from plotting.plotUncertainty import PlotUncertainty
from utils import create_logger
from models import Model
from uncertainpy import Data
from parameters import Parameters


class UncertaintyEstimation(object):
    def __init__(self,
                 model,
                 parameters,
                 features=None,
                 base_features=GeneralFeatures,
                 save_figures=True,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 save_data=True,
                 output_dir_data="data/",
                 xlabel=None,
                 ylabel=None,
                 verbose_level="info",
                 verbose_filename=None,
                 uncertainty_calculations=None,
                 PCECustom=None,
                 CPUs=mp.cpu_count(),
                 supress_model_graphics=True,
                 M=3,
                 nr_pc_samples=None,
                 nr_mc_samples=10*3,
                 nr_pc_mc_samples=10*5,
                 seed=None):

        self.data = None

        self._model = None
        self._features = None
        self._parameters = None

        self.base_features = base_features

        self.save_figures = save_figures
        self.save_data = save_data
        self.output_dir_data = output_dir_data
        self.output_dir_figures = output_dir_figures

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        if uncertainty_calculations is None:
            self.uncertainty_calculations = UncertaintyCalculations(
                model,
                features=features,
                CPUs=CPUs,
                supress_model_graphics=supress_model_graphics,
                M=M,
                nr_pc_samples=nr_pc_samples,
                nr_mc_samples=nr_mc_samples,
                nr_pc_mc_samples=nr_pc_mc_samples,
                seed=seed,
                verbose_level=verbose_level,
                verbose_filename=verbose_filename
            )
        else:
            self.uncertainty_calculations = uncertainty_calculations

        if PCECustom is not None:
            self.uncertainty_calculations.PCECustom = types.MethodType(PCECustom,
                                                                       self.uncertainty_calculations)



        self.features = features
        self.parameters = parameters
        self.model = model


        if xlabel is not None:
            self.model.xlabel = xlabel

        if ylabel is not None:
            self.model.ylabel = ylabel


        self.plotting = PlotUncertainty(output_dir=self.output_dir_figures,
                                        figureformat=figureformat,
                                        verbose_level=verbose_level,
                                        verbose_filename=verbose_filename)




    @property
    def features(self):
        return self._features


    @features.setter
    def features(self, new_features):
        if new_features is None:
            self._features = self.base_features(features_to_run=None)
        elif isinstance(new_features, GeneralFeatures):
            self._features = new_features
        else:
            self._features = self.base_features(features_to_run="all")
            self._features.add_features(new_features)
            self._features.features_to_run = "all"

        self.uncertainty_calculations.features = self.features


    @property
    def parameters(self):
        return self._parameters


    @parameters.setter
    def parameters(self, new_parameters):
        if isinstance(new_parameters, Parameters) or new_parameters is None:
            self._parameters = new_parameters
        else:
            self._parameters = Parameters(new_parameters)

        self.uncertainty_calculations.parameters = new_parameters


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        if isinstance(new_model, Model):
            self._model = new_model
        elif callable(new_model):
            self._model = Model()
            self._model.run = new_model
        else:
            raise TypeError("model must be a Model instance or callable")

        self.uncertainty_calculations.model = new_model


    @property
    def uncertainty_calculations(self):
        return self._uncertainty_calculations


    @uncertainty_calculations.setter
    def uncertainty_calculations(self, new_uncertainty_calculations):
        self._uncertainty_calculations = new_uncertainty_calculations

        self.uncertainty_calculations.features = self.features
        self.uncertainty_calculations.model = self.model


    # TODO add features_to_run as argument to this function
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
            self.CustomUQ(plot_condensed=plot_condensed,
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
            if filename is None:
                filename = self.model.name

            self.save(filename, output_dir=output_dir_data)


        if self.save_figures:
            self.plot(condensed=plot_condensed,
                      sensitivity=False,
                      output_dir=output_dir_figures)

        if plot_simulator_results:
            self.plot(simulator_results=True, output_dir=output_dir_figures)



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
            raise RuntimeWarning("The number of uncertain parameters is high."
                                 + "A Monte-Carlo method _might_ be faster.")


        self.data = self.uncertainty_calculations.PC(
            uncertain_parameters=uncertain_parameters,
            method=method,
            rosenblatt=rosenblatt
        )

        if self.save_data:
            if filename is None:
                filename = self.model.name

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
                filename = self.model.name

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
            raise RuntimeWarning("The number of uncertain parameters is high. "
                                 + "A Monte-Carlo method _might_ be faster.")


        if filename is None:
            filename = self.model.name


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
            filename = self.model.name


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


        self.plotting.data = self.data
        self.plotting.output_dir = output_dir

        self.logger.info("Plotting in {}".format(output_dir))

        if simulator_results:
            self.plotting.plotSimulatorResults()
        else:
            self.plotting.plot(condensed=condensed,
                               sensitivity=sensitivity)



    def convertUncertainParameters(self, uncertain_parameters):
        if uncertain_parameters is None:
            uncertain_parameters = self.parameters.getUncertain("name")

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        return uncertain_parameters
