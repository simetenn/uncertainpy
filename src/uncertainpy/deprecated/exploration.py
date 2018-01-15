import time
import os

from .uncertainty_calculations import UncertaintyCalculations
from .features import Features
from .plotting import PlotUncertainty
from .utils import create_logger
from .data import Data


from .uncertainty import UncertaintyQuantification
from .distribution import uniform, normal
from uncertainpy.plotting import PlotUncertaintyCompare
from uncertainpy.utils import create_logger



# TODO DOES NOT WORK YET!

class UncertaintyQuantifications():
    def __init__(self, model,
                 features=None,
                 uncertainty_calculations=None,
                 plot=False,
                 plot_type="results",
                 plot_model=False,
                 figure_folder="figures/",
                 figureformat=".png",
                 save_data=True,
                 data_folder="data/",
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
        # figure_folder/distribution_interval/parameter_value-that-is-plotted.figure-format

        self.data = None

        if features is None:
            self.features = Features(features_to_run=None)
        else:
            self.features = features

        self.plot = plot
        self.save_data = save_data
        self.data_folder = data_folder
        self.figure_folder = figure_folder
        self.plot_model = plot_model

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        if features is None:
            self.features = Features(features_to_run=None)
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

        self.plotting = PlotUncertainty(output_dir=self.figure_folder,
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
                current_figure_folder = os.path.join(self.figure_folder,
                                                          distribution_function + "_%g" % interval)

                # distribution = getattr(Distribution(interval), distribution_function)
                distribution = distributions(interval)
                self.model.set_all_distributions(distribution)

                message = "Running for: " + distribution_function + " " + str(interval)
                self.logger.info(message)

                tmp_data_folder = \
                    os.path.join(self.data_folder,
                                 distribution_function + "_%g" % interval)

                self.uncertainty_estimations =\
                    UncertaintyQuantification(self.model,
                                          features=self.features,
                                          plot=self.plot,
                                          figure_folder=current_figure_folder,
                                          figureformat=self.figureformat,
                                          save_data=self.save_data,
                                          data_folder=tmp_data_folder,
                                          output_data_filename=self.model.__class__.__name__,
                                          suppress_model_graphics=self.suppress_model_graphics,
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

                if self.plot_model:
                    self.uncertainty_estimations.results()

                del self.uncertainty_estimations



    def comparemonte_carlo(self, nr_mc_samples):
        run_times = []

        name = "pc"
        figure_folder = os.path.join(self.figure_folder, name)
        data_folder = os.path.join(self.data_folder, name)

        compare_folders = [name]

        self.uncertainty_estimations =\
            UncertaintyQuantification(self.model,
                                  features=self.features,
                                  plot=self.plot,
                                  figure_folder=figure_folder,
                                  figureformat=self.figureformat,
                                  save_data=self.save_data,
                                  data_folder=data_folder,
                                  output_data_filename=self.model.__class__.__name__,
                                  suppress_model_graphics=self.suppress_model_graphics,
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

        if self.plot_model:
            self.uncertainty_estimations.results()


        del self.uncertainty_estimations

        run_times.append(time.time() - time_1)

        self.mc_var = {}
        for nr_mc_sample in nr_mc_samples:
            message = "Running for: " + str(nr_mc_sample)
            self.logger.info(message)

            name = "mc_" + str(nr_mc_sample)
            current_figure_folder = os.path.join(self.figure_folder, name)
            tmp_data_folder = os.path.join(self.data_folder, name)

            compare_folders.append(name)

            self.uncertainty_estimations =\
                UncertaintyQuantification(self.model,
                                      features=self.features,
                                      plot=self.plot,
                                      figure_folder=current_figure_folder,
                                      figureformat=self.figureformat,
                                      save_data=self.save_data,
                                      data_folder=tmp_data_folder,
                                      output_data_filename=self.model.__class__.__name__,
                                      suppress_model_graphics=self.suppress_model_graphics,
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

            self.uncertainty_estimations.allParametersmonte_carlo()
            if self.plot_model:
                self.uncertainty_estimations.results()

            # self.mc_var[nr_mc_sample] = self.uncertainty_estimations.Var

            del self.uncertainty_estimations

            run_times.append(time.time() - time_1)


        if self.plot:
            plot = PlotUncertaintyCompare(data_dir=self.data_folder,
                                          figure_folder=self.figure_folder,
                                          figureformat=self.figureformat,
                                          verbose_level=self.verbose_level,
                                          verbose_filename=self.verbose_filename,
                                          xlabel=self.model.xlabel,
                                          ylabel=self.model.ylabel)

            plot.plotCompareAll(self.output_data_filename, compare_folders=compare_folders)

        return run_times



    def timePassed(self):
        return time.time() - self.t_start
