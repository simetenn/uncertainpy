import time
import os

from uncertainty_calculations import UncertaintyCalculations
from uncertainpy.features import GeneralFeatures
from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.utils import create_logger
from uncertainpy import Data


from uncertainpy import UncertaintyEstimation, Distribution
from uncertainpy.plotting import PlotUncertaintyCompare
from uncertainpy.utils import create_logger



# TODO DOES NOT WORK YET!

class UncertaintyEstimations():
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
                 verbose_filename=None,
                 **kwargs):
        """
        Options can also be sent to the feature
        kwargs:
        feature_options = {keyword1: value1, keyword2: value2}
        """

        # Figures are always saved on the format:
        # output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-format

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







    def exploreParameters(self, distributions):
        for distribution_function in distributions:
            for interval in distributions[distribution_function]:
                current_output_dir_figures = os.path.join(self.output_dir_figures,
                                                          distribution_function + "_%g" % interval)

                distribution = getattr(Distribution(interval), distribution_function)

                self.model.setAllDistributions(distribution)

                message = "Running for: " + distribution_function + " " + str(interval)
                self.logger.info(message)

                tmp_output_dir_data = \
                    os.path.join(self.output_dir_data,
                                 distribution_function + "_%g" % interval)

                self.uncertainty_estimations =\
                    UncertaintyEstimation(self.model,
                                          features=self.features,
                                          save_figures=self.save_figures,
                                          output_dir_figures=current_output_dir_figures,
                                          figureformat=self.figureformat,
                                          save_data=self.save_data,
                                          output_dir_data=tmp_output_dir_data,
                                          output_data_filename=self.model.__class__.__name__,
                                          supress_model_graphics=self.supress_model_graphics,
                                          supress_model_output=self.supress_model_output,
                                          CPUs=self.CPUs,
                                          rosenblatt=self.rosenblatt,
                                          nr_mc_samples=self.nr_mc_samples,
                                          nr_pc_mc_samples=self.nr_pc_mc_samples,
                                          verbose_level=self.verbose_level,
                                          verbose_filename=self.verbose_filename,
                                          seed=self.seed,
                                          **self.kwargs)

                if self.single_parameter_runs:
                    self.uncertainty_estimations.singleParameters()
                self.uncertainty_estimations.allParameters()

                if self.plot_simulator_results:
                    self.uncertainty_estimations.plotSimulatorResults()

                del self.uncertainty_estimations



    def compareMC(self, nr_mc_samples):
        run_times = []

        name = "pc"
        output_dir_figures = os.path.join(self.output_dir_figures, name)
        output_dir_data = os.path.join(self.output_dir_data, name)

        compare_folders = [name]

        self.uncertainty_estimations =\
            UncertaintyEstimation(self.model,
                                  features=self.features,
                                  save_figures=self.save_figures,
                                  output_dir_figures=output_dir_figures,
                                  figureformat=self.figureformat,
                                  save_data=self.save_data,
                                  output_dir_data=output_dir_data,
                                  output_data_filename=self.model.__class__.__name__,
                                  supress_model_graphics=self.supress_model_graphics,
                                  supress_model_output=self.supress_model_output,
                                  CPUs=self.CPUs,
                                  rosenblatt=self.rosenblatt,
                                  nr_pc_mc_samples=self.nr_pc_mc_samples,
                                  verbose_level=self.verbose_level,
                                  verbose_filename=self.verbose_filename,
                                  seed=self.seed,
                                  **self.kwargs)

        time_1 = time.time()

        self.uncertainty_estimations.allParameters()

        if self.plot_simulator_results:
            self.uncertainty_estimations.plotSimulatorResults()


        del self.uncertainty_estimations

        run_times.append(time.time() - time_1)

        self.mc_var = {}
        for nr_mc_sample in nr_mc_samples:
            message = "Running for: " + str(nr_mc_sample)
            self.logger.info(message)

            name = "mc_" + str(nr_mc_sample)
            current_output_dir_figures = os.path.join(self.output_dir_figures, name)
            tmp_output_dir_data = os.path.join(self.output_dir_data, name)

            compare_folders.append(name)

            self.uncertainty_estimations =\
                UncertaintyEstimation(self.model,
                                      features=self.features,
                                      save_figures=self.save_figures,
                                      output_dir_figures=current_output_dir_figures,
                                      figureformat=self.figureformat,
                                      save_data=self.save_data,
                                      output_dir_data=tmp_output_dir_data,
                                      output_data_filename=self.model.__class__.__name__,
                                      supress_model_graphics=self.supress_model_graphics,
                                      supress_model_output=self.supress_model_output,
                                      CPUs=self.CPUs,
                                      rosenblatt=self.rosenblatt,
                                      nr_mc_samples=nr_mc_sample,
                                      nr_pc_mc_samples=self.nr_pc_mc_samples,
                                      verbose_level=self.verbose_level,
                                      verbose_filename=self.verbose_filename,
                                      seed=self.seed,
                                      **self.kwargs)


            time_1 = time.time()

            self.uncertainty_estimations.allParametersMC()
            if self.plot_simulator_results:
                self.uncertainty_estimations.plotSimulatorResults()

            # self.mc_var[nr_mc_sample] = self.uncertainty_estimations.Var

            del self.uncertainty_estimations

            run_times.append(time.time() - time_1)


        if self.save_figures:
            plot = PlotUncertaintyCompare(data_dir=self.output_dir_data,
                                          output_dir_figures=self.output_dir_figures,
                                          figureformat=self.figureformat,
                                          verbose_level=self.verbose_level,
                                          verbose_filename=self.verbose_filename,
                                          xlabel=self.model.xlabel,
                                          ylabel=self.model.ylabel)

            plot.plotCompareAll(self.output_data_filename, compare_folders=compare_folders)

        return run_times



    def timePassed(self):
        return time.time() - self.t_start
