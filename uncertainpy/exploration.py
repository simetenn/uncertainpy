import time
import os

import matplotlib.pyplot as plt
import multiprocessing as mp

from uncertainpy import UncertaintyEstimation, Distribution
from uncertainpy import prettyPlot, create_logger


class UncertaintyEstimations():
    def __init__(self, model,
                 feature_list=[],
                 features=None,
                 save_figures=False,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 save_data=True,
                 output_dir_data="data/",
                 plot_simulator_results=False,
                 supress_model_graphics=True,
                 supress_model_output=True,
                 CPUs=mp.cpu_count(),
                 rosenblatt=False,
                 nr_mc_samples=10**3,
                 nr_pc_mc_samples=10**5,
                 verbose_level="error",
                 verbose_filename=None,
                 seed=None,
                 **kwargs):
        """
        Options can also be sent to the feature
        kwargs:
        feature_options = {keyword1: value1, keyword2: value2}
        """

        # Figures are always saved on the format:
        # output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-format



        self.uncertainty_estimations = None

        # original_parameters, uncertain_parameters, distributions,

        self.model = model

        self.save_figures = save_figures
        self.output_dir_figures = output_dir_figures
        self.save_data = save_data
        self.output_dir_data = output_dir_data
        self.plot_simulator_results = plot_simulator_results

        self.supress_model_graphics = supress_model_graphics
        self.supress_model_output = supress_model_output
        self.CPUs = CPUs
        self.rosenblatt = rosenblatt
        self.figureformat = figureformat
        self.features = features
        self.feature_list = feature_list
        self.nr_mc_samples = nr_mc_samples
        self.nr_pc_mc_samples = nr_pc_mc_samples

        self.kwargs = kwargs

        self.t_start = time.time()

        if not os.path.isdir(output_dir_data):
            os.makedirs(output_dir_data)

        self.verbose_level = verbose_level
        self.verbose_filename = verbose_filename

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        self.seed = seed



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
                                          feature_list=self.feature_list,
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



        self.uncertainty_estimations =\
            UncertaintyEstimation(self.model,
                                  feature_list=self.feature_list,
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

        # self.pc_var = self.uncertainty_estimations.Var
        # self.t_pc = self.uncertainty_estimations.t
        # nr_pc_samples = self.uncertainty_estimations.nr_pc_samples
        # features_2d = self.uncertainty_estimations.features_2d
        # features_1d = self.uncertainty_estimations.features_1d

        del self.uncertainty_estimations

        run_times.append(time.time() - time_1)

        self.mc_var = {}
        for nr_mc_sample in nr_mc_samples:
            message = "Running for: " + str(nr_mc_sample)
            self.logger.info(message)

            name = "mc_" + str(nr_mc_sample)
            current_output_dir_figures = os.path.join(self.output_dir_figures, name)
            tmp_output_dir_data = os.path.join(self.output_dir_data, name)

            self.uncertainty_estimations =\
                UncertaintyEstimation(self.model,
                                      feature_list=self.feature_list,
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


        return run_times


    def plot_compareMC_1d(self, feature):
        if feature not in self.features_1d:
            # TODO is this the right error to raise?
            raise RuntimeError("%s is not a 1D feature" % (feature))


        output_dir_compare = os.path.join(self.output_dir_figures, "MC-compare")
        if not os.path.isdir(output_dir_compare):
            os.makedirs(output_dir_compare)

        color = 0
        max_var = 0
        min_var = 10**10
        legend = []
        new_figure = True

        for nr_mc_sample in sorted(self.mc_var):

            if self.mc_var[nr_mc_sample]["directComparison"].max() > max_var:
                max_var = self.mc_var[nr_mc_sample]["directComparison"].max()

            if self.mc_var[nr_mc_sample]["directComparison"].min() < min_var:
                min_var = self.mc_var[nr_mc_sample]["directComparison"].min()

            legend.append("MC samples " + str(nr_mc_sample))

            prettyPlot(self.t_pc["directComparison"], self.mc_var[nr_mc_sample]["directComparison"],
                       new_figure=new_figure, color=color,
                       xlabel="Time", ylabel="Variance, mv",
                       title="Variance")
            new_figure = False
            color += 2


        if self.pc_var["directComparison"].max() > max_var:
            max_var = self.pc_var["directComparison"].max()

        if self.pc_var["directComparison"].min() < min_var:
            min_var = self.pc_var["directComparison"].min()

        legend.append("PC")

        prettyPlot(self.t_pc["directComparison"], self.pc_var["directComparison"],
                   new_figure=new_figure, color=color,
                   xlabel="Time", ylabel="Variance, mv",
                   title="Variance")
        new_figure = False
        color += 2

        plt.ylim([min_var*0.99, max_var*1.01])
        plt.legend(legend)
        plt.show()
        plt.savefig(os.path.join(output_dir_compare,
                                 "variance-MC-PC_" + self.figureformat))
        # plt.show()
        plt.close()



    def timePassed(self):
        return time.time() - self.t_start
