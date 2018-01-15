import os
import types
import multiprocess as mp
import numpy as np

from .core.uncertainty_calculations import UncertaintyCalculations
from .plotting.plot_uncertainty import PlotUncertainty
from .utils import create_logger
from .data import Data
from .core.base import ParameterBase


#  Figures are always saved on the format:
#  output_dir_figures/parameter_value-that-is-plotted.figure-extension

class UncertaintyQuantification(ParameterBase):
    def __init__(self,
                 model,
                 parameters,
                 features=None,
                 uncertainty_calculations=None,
                 verbose_level="info",
                 verbose_filename=None,
                 create_PCE_custom=None,
                 CPUs=mp.cpu_count(),
                 suppress_model_graphics=True):


        if uncertainty_calculations is None:
            self._uncertainty_calculations = UncertaintyCalculations(
                model=model,
                parameters=parameters,
                features=features,
                CPUs=CPUs,
                suppress_model_graphics=suppress_model_graphics,
                verbose_level=verbose_level,
                verbose_filename=verbose_filename
            )
        else:
            self._uncertainty_calculations = uncertainty_calculations

        super(UncertaintyQuantification, self).__init__(parameters=parameters,
                                                        model=model,
                                                        features=features,
                                                        verbose_level=verbose_level,
                                                        verbose_filename=verbose_filename)



        self.data = None

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        if create_PCE_custom is not None:
            self.uncertainty_calculations.create_PCE_custom = types.MethodType(create_PCE_custom,
                                                                               self.uncertainty_calculations)


        self.plotting = PlotUncertainty(verbose_level=verbose_level,
                                        verbose_filename=verbose_filename)


    @ParameterBase.features.setter
    def features(self, new_features):
        ParameterBase.features.fset(self, new_features)

        self.uncertainty_calculations.features = self.features


    @ParameterBase.model.setter
    def model(self, new_model):
        ParameterBase.model.fset(self, new_model)

        self.uncertainty_calculations.model = self.model


    @ParameterBase.parameters.setter
    def parameters(self, new_parameters):
        ParameterBase.parameters.fset(self, new_parameters)

        self.uncertainty_calculations.parameters = self.parameters


    @property
    def uncertainty_calculations(self):
        return self._uncertainty_calculations


    @uncertainty_calculations.setter
    def uncertainty_calculations(self, new_uncertainty_calculations):
        self._uncertainty_calculations = new_uncertainty_calculations

        self.uncertainty_calculations.features = self.features
        self.uncertainty_calculations.model = self.model


    # TODO add features_to_run as argument to this function
    def quantify(self,
                 method="pc",
                 pc_method="collocation",
                 rosenblatt=False,
                 uncertain_parameters=None,
                 single=False,
                 polynomial_order=3,
                 nr_collocation_nodes=None,
                 quadrature_order=4,
                 nr_pc_mc_samples=10**4,
                 nr_mc_samples=10**3,
                 allow_incomplete=False,
                 seed=None,
                 plot_condensed=True,
                 plot_results=False,
                 output_dir_figures="figures",
                 output_dir_data="data",
                 filename=None,
                 sensitivity="sensitivity_1",
                 **custom_kwargs):
        """
method: pc, mc
pc_method: "collocation, spectral"
        """
        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)


        if method.lower() == "pc":
            if single:
                self.polynomial_chaos_single(uncertain_parameters=uncertain_parameters,
                                             method=pc_method,
                                             rosenblatt=rosenblatt,
                                             polynomial_order=polynomial_order,
                                             nr_collocation_nodes=nr_collocation_nodes,
                                             quadrature_order=quadrature_order,
                                             nr_pc_mc_samples=nr_pc_mc_samples,
                                             allow_incomplete=allow_incomplete,
                                             seed=seed,
                                             plot_condensed=plot_condensed,
                                             plot_results=plot_results,
                                             output_dir_figures=output_dir_figures,
                                             output_dir_data=output_dir_data,
                                             filename=filename,
                                             **custom_kwargs)
            else:
                self.polynomial_chaos(uncertain_parameters=uncertain_parameters,
                                      method=pc_method,
                                      rosenblatt=rosenblatt,
                                      polynomial_order=polynomial_order,
                                      nr_collocation_nodes=nr_collocation_nodes,
                                      quadrature_order=quadrature_order,
                                      nr_pc_mc_samples=nr_pc_mc_samples,
                                      allow_incomplete=allow_incomplete,
                                      seed=seed,
                                      plot_condensed=plot_condensed,
                                      plot_results=plot_results,
                                      output_dir_figures=output_dir_figures,
                                      output_dir_data=output_dir_data,
                                      filename=filename,
                                      sensitivity=sensitivity,
                                      **custom_kwargs)
        elif method.lower() == "mc":
            if single:
                self.monte_carlo_single(uncertain_parameters=uncertain_parameters,
                                        nr_samples=nr_mc_samples,
                                        plot_condensed=plot_condensed,
                                        plot_results=plot_results,
                                        output_dir_figures=output_dir_figures,
                                        output_dir_data=output_dir_data,
                                        filename=filename,
                                        seed=seed)
            else:
                self.monte_carlo(uncertain_parameters=uncertain_parameters,
                                 nr_samples=nr_mc_samples,
                                 plot_condensed=plot_condensed,
                                 plot_results=plot_results,
                                 output_dir_figures=output_dir_figures,
                                 output_dir_data=output_dir_data,
                                 filename=filename,
                                 seed=seed)

        elif method.lower() == "custom":
            self.custom_uncertainty_quantification(plot_condensed=plot_condensed,
                                                   plot_results=plot_results,
                                                   output_dir_figures=output_dir_figures,
                                                   output_dir_data=output_dir_data,
                                                   filename=filename,
                                                   **custom_kwargs)

        else:
            raise ValueError("No method with name {}".format(method))



    def custom_uncertainty_quantification(self,
                                          save_figures=True,
                                          plot_condensed=True,
                                          plot_results=False,
                                          output_dir_figures="figures",
                                          output_dir_data="data",
                                          filename=None,
                                          **custom_kwargs):


        self.data = self.uncertainty_calculations.custom_uncertainty_quantification(**custom_kwargs)

        if filename is None:
            filename = self.model.name

        self.save(filename, output_dir=output_dir_data)


        if save_figures:
            self.plot(condensed=plot_condensed,
                      sensitivity=None,
                      output_dir=output_dir_figures)

        if plot_results:
            self.plot(results=True, output_dir=output_dir_figures)



    def polynomial_chaos(self,
                         uncertain_parameters=None,
                         method="collocation",
                         rosenblatt=False,
                         polynomial_order=3,
                         nr_collocation_nodes=None,
                         quadrature_order=4,
                         nr_pc_mc_samples=10**4,
                         allow_incomplete=False,
                         seed=None,
                         plot_condensed=True,
                         plot_results=False,
                         save_figures=True,
                         output_dir_figures="figures",
                         output_dir_data="data",
                         sensitivity="sensitivity_1",
                         filename=None,
                         **custom_kwargs):

        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high."
                                 + "The Monte-Carlo method might be faster.")


        self.data = self.uncertainty_calculations.polynomial_chaos(
            uncertain_parameters=uncertain_parameters,
            method=method,
            rosenblatt=rosenblatt,
            polynomial_order=polynomial_order,
            nr_collocation_nodes=nr_collocation_nodes,
            quadrature_order=quadrature_order,
            nr_pc_mc_samples=nr_pc_mc_samples,
            allow_incomplete=allow_incomplete,
            seed=seed,
            **custom_kwargs
        )

        if filename is None:
            filename = self.model.name

        self.save(filename, output_dir=output_dir_data)


        if save_figures:
            self.plot(condensed=plot_condensed,
                      output_dir=output_dir_figures,
                      sensitivity=sensitivity)

        if plot_results:
            self.plot(results=True, output_dir=output_dir_figures)


    def monte_carlo(self,
                    uncertain_parameters=None,
                    nr_samples=10**3,
                    plot_condensed=True,
                    plot_results=False,
                    output_dir_figures="figures",
                    output_dir_data="data",
                    filename=None,
                    save_figures=True,
                    seed=None):

        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        self.data = self.uncertainty_calculations.monte_carlo(uncertain_parameters=uncertain_parameters,
                                                              nr_samples=nr_samples,
                                                              seed=seed)


        if filename is None:
           filename = self.model.name

        self.save(filename, output_dir=output_dir_data)


        if save_figures:
            self.plot(condensed=plot_condensed,
                      sensitivity=None,
                      output_dir=output_dir_figures)

        if plot_results:
            self.plot(results=True, output_dir=output_dir_figures)



    def polynomial_chaos_single(self,
                                uncertain_parameters=None,
                                method="collocation",
                                rosenblatt=False,
                                polynomial_order=3,
                                nr_collocation_nodes=None,
                                quadrature_order=4,
                                nr_pc_mc_samples=10**4,
                                allow_incomplete=False,
                                seed=None,
                                plot_condensed=True,
                                plot_results=False,
                                output_dir_data="figures",
                                output_dir_figures="data",
                                filename=None,
                                save_figures=True):

        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high. "
                                 + "A Monte Carlo method might be faster.")

        if filename is None:
            filename = self.model.name

        if seed is not None:
            np.random.seed(seed)


        for uncertain_parameter in uncertain_parameters:
            self.logger.info("Running for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.polynomial_chaos(
                uncertain_parameters=uncertain_parameter,
                method=method,
                rosenblatt=rosenblatt,
                polynomial_order=polynomial_order,
                nr_collocation_nodes=nr_collocation_nodes,
                quadrature_order=quadrature_order,
                nr_pc_mc_samples=nr_pc_mc_samples,
                allow_incomplete=allow_incomplete,
            )

            tmp_filename = "{}_single-parameter-{}".format(
                filename,
                uncertain_parameter
            )

            self.save(tmp_filename, output_dir=output_dir_data)

            tmp_output_dir_figures = os.path.join(output_dir_figures, tmp_filename)

            if save_figures:
                self.plot(condensed=plot_condensed,
                          sensitivity=None,
                          output_dir=tmp_output_dir_figures)

            if plot_results:
                self.plot(results=True,
                          output_dir=tmp_output_dir_figures)


    def monte_carlo_single(self,
                           uncertain_parameters=None,
                           nr_samples=10**3,
                           plot_condensed=True,
                           plot_results=False,
                           output_dir_data="data",
                           output_dir_figures="figures",
                           save_figures=True,
                           filename=None,
                           seed=None):
        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        if filename is None:
            filename = self.model.name

        if seed is not None:
            np.random.seed(seed)

        for uncertain_parameter in uncertain_parameters:
            self.logger.info("Running MC for " + uncertain_parameter)

            self.data = self.uncertainty_calculations.monte_carlo(uncertain_parameters=uncertain_parameter,
                                                                  nr_samples=nr_samples)

            tmp_filename = "{}_single-parameter-{}".format(
                filename,
                uncertain_parameter
            )

            self.save(tmp_filename, output_dir=output_dir_data)

            tmp_output_dir_figures = os.path.join(output_dir_figures, tmp_filename)

            if save_figures:
                self.plot(condensed=plot_condensed, output_dir=tmp_output_dir_figures)

            if plot_results:
                self.plot(results=True, output_dir=tmp_output_dir_figures)



    def save(self, filename, output_dir="data"):

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_path = os.path.join(output_dir, filename + ".h5")

        self.logger.info("Saving data as: {}".format(save_path))

        self.data.save(save_path)



    def load(self, filename):
        self.data = Data(filename)


    def plot(self,
             condensed=True,
             sensitivity="sensitivity_1",
             results=False,
             output_dir="figures",
             figureformat=".png"):

        self.plotting.data = self.data
        self.plotting.output_dir = output_dir
        self.plotting.figureformat = ".png"

        if results:
            self.plotting.all_results()
        else:
            self.plotting.plot(condensed=condensed,
                               sensitivity=sensitivity)
