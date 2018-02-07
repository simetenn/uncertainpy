import glob
# import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from .prettyplot import prettyPlot, prettyBar
from .prettyplot import spines_color, get_current_colormap
from .prettyplot import get_colormap_tableu20, set_style
from .prettyplot import axis_grey, labelsize, fontsize, titlesize, linewidth

from ..data import Data
from ..utils import create_logger, str_to_latex, list_to_latex


# TODO find a good way to find the directory where the data files are

# TODO compare plots in a grid of all plots,
# such as plotting all features in a grid plot

# TODO Change the use of **plot_kwargs to use a dict for specific plotting commands?



class PlotUncertainty(object):
    """
    Plotting the results from the uncertainty quantification and sensitivity
    analysis.

    Parameters
    ----------
    filename : {None, str}, optional
        The name of the data file. If given the file is loaded.
        If None, nothing is loaded.
        Default is None.
    folder : str, optional
        The folder where to save the plots. Creates a new folder if it does not
        exist.
        Default is "figures/".
    figureformat : str, optional
        The format to save the plots in. Given as ".xxx". All formats
        supported by Matplotlib are available.
        Default is ".png",
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if set. Default is None.

    Attributes
    ----------
    folder : str
        The folder where to save the plots.
    figureformat : str, optional
        The format to save the plots in. Given as ".xxx". All formats
        supported by Matplotlib are available.
    data : Data
        A data object that contains the results from the uncertainty quantification.
        Contains all model and feature values, as well as all calculated
        statistical metrics.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.
    """
    def __init__(self,
                 filename=None,
                 folder="figures/",
                 figureformat=".png",
                 verbose_level="info",
                 verbose_filename=None):

        self._folder = None

        self.folder = folder
        self.figureformat = figureformat

        self.features_in_combined_plot = 3

        self.data = None

        if filename is not None:
            self.load(filename)


        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)



    def load(self, filename):
        self.data = Data(filename)


    @property
    def folder(self):
        return self._folder


    @folder.setter
    def folder(self, new_folder):
        self._folder = new_folder

        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)




    def all_evaluations(self, foldername="evaluations"):
        for feature in self.data:
            self.evaluations(feature=feature, foldername=foldername)


    def evaluations(self, feature=None, foldername=""):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        save_folder = os.path.join(self.folder, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        dimension = self.data.ndim(feature)
        if dimension == 0:
            self.evaluations_0d(feature=feature, foldername=foldername)

        elif dimension == 1:
            self.evaluations_1d(feature=feature, foldername=foldername)

        elif dimension == 2:
            self.evaluations_2d(feature=feature, foldername=foldername)

        elif dimension > 2:
            raise NotImplementedError(">2D plots not implemented.")
        else:
            raise AttributeError("Dimension of evaluations is not valid: dim {}".format(dimension))



    def evaluations_0d(self, feature=None, foldername="", **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data or "evaluations"  not in self.data[feature]:
            self.logger.warning("No {} evaluations to plot.".format(feature))
            return

        if self.data.ndim(feature) != 0:
            raise ValueError("{} is not a 0D feature".format(feature))

        save_folder = os.path.join(self.folder, foldername, feature + "_evaluations")
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        prettyPlot(self.data[feature].evaluations,
                   xlabel=r"evaluation #number",
                   ylabel=self.data.get_labels(feature)[0],
                   title="{}, evaluations".format(feature.replace("_", " ")),
                   new_figure=True,
                   **plot_kwargs)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "evaluations" + self.figureformat))
        plt.close()


    def evaluations_1d(self, feature=None, foldername="", **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data or "evaluations"  not in self.data[feature]:
            self.logger.warning("No model evaluations to plot.")
            return

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        i = 1
        save_folder = os.path.join(self.folder, foldername, feature + "_evaluations")
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature].evaluations[0]))
        else:
            time = self.data[feature].time


        padding = len(str(len(self.data[feature].evaluations[0]) + 1))
        for evaluation in self.data[feature].evaluations:
            ax = prettyPlot(time, evaluation,
                            xlabel=xlabel, ylabel=ylabel,
                            title="{}, evaluation {:d}".format(feature.replace("_", " "), i), new_figure=True, **plot_kwargs)
            ax.set_xlim([min(time), max(time)])
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder,
                                     "evaluation_{0:0{1}d}".format(i, padding) + self.figureformat))
            plt.close()
            i += 1



    def evaluations_2d(self, feature=None, foldername="", **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data or "evaluations" not in self.data[feature]:
            self.logger.warning("No model evaluations to plot.")
            return

        if self.data.ndim(feature) != 2:
            raise ValueError("{} is not a 2D feature.".format(feature))

        set_style("seaborn-dark")

        save_folder = os.path.join(self.folder, foldername, feature + "_evaluations")
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        labels = self.data.get_labels(feature)
        xlabel, ylabel, zlabel = labels

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature].evaluations[0]))
        else:
            time = self.data[feature].time

        padding = len(str(len(self.data[feature].evaluations) + 1))
        for i, evaluation in enumerate(self.data[feature].evaluations):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("{}, evaluation {:d}".format(feature.replace("_", " "), i))

            iax = ax.imshow(evaluation, cmap="viridis", aspect="auto",
                            extent=[time[0],
                                    time[-1],
                                    0, evaluation.shape[0]],
                            **plot_kwargs)

            cbar = fig.colorbar(iax)
            cbar.ax.set_ylabel(zlabel)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder,
                                     "evaluation_{0:0{1}d}".format(i, padding) + self.figureformat))
            plt.close()




    def attribute_feature_1d(self,
                             feature=None,
                             attribute="mean",
                             attribute_name="mean",
                             hardcopy=True,
                             show=False,
                             **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if attribute not in ["mean", "variance"]:
            raise ValueError("{} is not a supported attribute".format(attribute))

        if attribute not in self.data[feature]:
            msg = " Unable to plot {attribute_name}. {attribute_name} of {feature} does not exist."
            self.logger.warning(msg.format(attribute_name=attribute, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][attribute]))
        else:
            time = self.data[feature].time


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature + ", " + attribute_name
        ax = prettyPlot(time, self.data[feature][attribute],
                        title.replace("_", " "), xlabel, ylabel, **plot_kwargs)

        ax.set_xlim([min(time), max(time)])

        save_name = feature + "_" + attribute_name
        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     save_name + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()



    def attribute_feature_2d(self,
                             feature=None,
                             attribute="mean",
                             attribute_name="mean",
                             hardcopy=True,
                             show=False,
                             **plot_kwargs):

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 2:
            raise ValueError("{} is not a 2D feature".format(feature))

        if attribute not in ["mean", "variance"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        if attribute not in self.data[feature]:
            msg = " Unable to plot {attribute_name}. {attribute_name} of {feature} does not exist."
            self.logger.warning(msg.format(attribute_name=attribute, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            extent = None
        else:
            extent=[self.data[feature].time[0], self.data[feature].time[-1],
                    0, self.data[feature][attribute].shape[0]]



        title = feature + ", " + attribute_name
        labels = self.data.get_labels(feature)
        xlabel, ylabel, zlabel = labels

        set_style("seaborn-dark")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title.replace("_", " "))


        iax = ax.imshow(self.data[feature][attribute], cmap="viridis", aspect="auto",
                        extent=extent,
                        **plot_kwargs)

        cbar = fig.colorbar(iax)
        # cbar.ax.set_title(zlabel)
        cbar.ax.set_ylabel(zlabel)

        ax.set_xlabel(xlabel, fontsize=labelsize)
        ax.set_ylabel(ylabel, fontsize=labelsize)

        save_name = feature + "_" + attribute_name

        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     save_name + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()



    def mean_1d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_1d(feature,
                                  attribute="mean",
                                  attribute_name="mean",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)


    def variance_1d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_1d(feature,
                                  attribute="variance",
                                  attribute_name="variance",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)

    def mean_2d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_2d(feature,
                                  attribute="mean",
                                  attribute_name="mean",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)


    def variance_2d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_2d(feature,
                                  attribute="variance",
                                  attribute_name="variance",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)


    def mean_variance_1d(self,
                         feature=None,
                         new_figure=True,
                         hardcopy=True,
                         show=False,
                         color=0,
                         style="seaborn-dark",
                         **plot_kwargs):


        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if "mean" not in self.data[feature] or "variance" not in self.data[feature]:
            msg = "Mean and/or variance of {feature} does not exist. ".format(feature=feature) \
                    + "Unable to plot mean and variance"
            self.logger.warning(msg)
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature].mean))
        else:
            time = self.data[feature].time


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels


        title = feature + ", mean and variance"
        ax = prettyPlot(time, self.data[feature].mean,
                        title.replace("_", " "), xlabel, ylabel + ", mean",
                        style=style, **plot_kwargs)


        colors = get_current_colormap()

        ax2 = ax.twinx()

        spines_color(ax2, edges={"top": "None", "bottom": "None",
                                 "right": colors[color+2], "left": "None"})
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=colors[color+2], labelcolor=colors[color+2], labelsize=labelsize)
        ax2.set_ylabel(ylabel + ', variance', color=colors[color+2], fontsize=labelsize)

        # ax2.set_ylim([min(self.data.variance[feature]), max(self.data.variance[feature])])

        ax2.plot(time, self.data[feature].variance,
                 color=colors[color+2], linewidth=linewidth, antialiased=True)

        ax2.yaxis.offsetText.set_fontsize(fontsize)
        ax2.yaxis.offsetText.set_color(colors[color+2])


        ax.tick_params(axis="y", color=colors[color], labelcolor=colors[color])
        ax.spines["left"].set_edgecolor(colors[color])
        ax.set_ylabel(ylabel + ', mean', color=colors[color], fontsize=labelsize)

        ax2.set_xlim([min(time), max(time)])
        ax.set_xlim([min(time), max(time)])


        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_mean-variance" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()
        #
        # if not show or not hardcopy:
        #     return ax, ax2



    def prediction_interval_1d(self,
                               feature=None,
                               hardcopy=True,
                               show=False,
                               **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if "mean" not in self.data[feature] \
            or "percentile_5" not in self.data[feature] \
                or "percentile_95" not in self.data[feature]:
            msg = "E, percentile_5  and/or percentile_95 of {feature} does not exist. Unable to plot prediction interval"
            self.logger.warning(msg.format(feature=feature))
            return


        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature].mean))
        else:
            time = self.data[feature].time


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature.replace("_", " ") + ", 90% prediction interval"
        ax = prettyPlot(time, self.data[feature].mean, title=title,
                        xlabel=xlabel, ylabel=ylabel, color=0,
                        **plot_kwargs)

        colors = get_current_colormap()
        ax.fill_between(time,
                         self.data[feature].percentile_5,
                         self.data[feature].percentile_95,
                         alpha=0.5, color=colors[0])

        ax.set_xlim([min(time), max(time)])
        plt.legend(["mean", "90% prediction interval"], loc="best")

        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_prediction-interval" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()


    def sensitivity_1d(self,
                       feature=None,
                       sensitivity="first",
                       hardcopy=True,
                       show=False,
                       **plot_kwargs):

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            time = self.data[feature].time

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        for i in range(len(self.data[feature][sensitivity])):
            ax = prettyPlot(time, self.data[feature][sensitivity][i],
                            title=title.capitalize() + ", " + feature.replace("_", " ") + " - " + self.data.uncertain_parameters[i],
                            xlabel=xlabel,
                            ylabel=title.capitalize(),
                            color=i,
                            nr_colors=len(self.data.uncertain_parameters), **plot_kwargs)
            # plt.ylim([0, 1.05])
            ax.set_xlim([min(time), max(time)])

            plt.tight_layout()

            if hardcopy:
                plt.savefig(os.path.join(self.folder,
                                         feature + "_" + sensitivity + "_"
                                         + self.data.uncertain_parameters[i] + self.figureformat))

            if show:
                plt.show()
            else:
                plt.close()




    def sensitivity_1d_grid(self,
                            feature=None,
                            sensitivity="first",
                            hardcopy=True,
                            show=False,
                            **plot_kwargs):

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title = self.convert_sensitivity(sensitivity)


        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            time = self.data[feature].time

        parameter_names = self.data.uncertain_parameters

        # get size of the grid in x and y directions
        nr_plots = len(parameter_names)
        grid_size = np.ceil(np.sqrt(nr_plots))
        grid_x_size = int(grid_size)
        grid_y_size = int(np.ceil(nr_plots/float(grid_x_size)))

        set_style("seaborn-darkgrid")
        fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size, squeeze=False, sharex='col', sharey='row')

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        # Add a larger subplot to use to set a common xlabel and ylabel
        set_style("seaborn-white")
        ax = fig.add_subplot(111, zorder=-10)
        spines_color(ax, edges={"top": "None", "bottom": "None",
                                "right": "None", "left": "None"})
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title.capitalize())

        for i in range(0, grid_x_size*grid_y_size):
            nx = i % grid_x_size
            ny = int(np.floor(i/float(grid_x_size)))

            ax = axes[ny][nx]

            if i < nr_plots:
                prettyPlot(time, self.data[feature][sensitivity][i],
                           title=parameter_names[i], color=i,
                           nr_colors=nr_plots, ax=ax,
                           **plot_kwargs)

                for tick in ax.get_xticklabels():
                    tick.set_rotation(-30)

                ax.set_ylim([0, 1.05])
                ax.set_xlim([min(time), max(time)])
                # ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)
                ax.tick_params(labelsize=10)
            else:
                ax.axis("off")

        title = title.capitalize() + ", " + feature.replace("_", " ")
        plt.suptitle(title, fontsize=titlesize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)


        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_" + sensitivity + "_grid" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()




    def sensitivity_1d_combined(self,
                                feature=None,
                                sensitivity="first",
                                hardcopy=True,
                                show=False,
                                **plot_kwargs):

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            time = self.data[feature].time


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        for i in range(len(self.data[feature][sensitivity])):
            prettyPlot(time,
                       self.data[feature][sensitivity][i],
                       title=title.capitalize() + ", " + feature.replace("_", " "),
                       xlabel=xlabel,
                       ylabel=title.capitalize(),
                       new_figure=False,
                       color=i,
                       nr_colors=len(self.data.uncertain_parameters),
                       label=self.data.uncertain_parameters[i],
                       **plot_kwargs)

        plt.ylim([0, 1.05])
        plt.xlim([min(time), max(time)])
        if len(self.data[feature][sensitivity]) > 4:
            plt.xlim([time[0], 1.3*time[-1]])

        plt.legend()
        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_" + sensitivity + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()


    def features_1d(self, sensitivity="first"):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total", None]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total or None, not {}".format(sensitivity))

        if sensitivity == "first":
            sensitivity = "sobol_first"
        elif sensitivity == "total":
            sensitivity = "sobol_total"


        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.mean_1d(feature=feature)
                self.variance_1d(feature=feature)
                self.mean_variance_1d(feature=feature)
                self.prediction_interval_1d(feature=feature)

                if sensitivity in self.data[feature]:
                    self.sensitivity_1d(feature=feature, sensitivity=sensitivity)
                    self.sensitivity_1d_combined(feature=feature, sensitivity=sensitivity)
                    self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity)



    def convert_sensitivity(self, sensitivity):
        if sensitivity == "first":
            sensitivity = "sobol_first"
        elif sensitivity == "total":
            sensitivity = "sobol_total"

        full_text = ""
        if sensitivity == "sobol_first":
            full_text = "first order Sobol indices"
        elif sensitivity == "sobol_total":
            full_text = "total order Sobol indices"

        return sensitivity, full_text


    def features_2d(self):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        for feature in self.data:
            if self.data.ndim(feature) == 2:
                self.mean_2d(feature=feature)
                self.variance_2d(feature=feature)


    # TODO not finished, missing correct label placement
    # TODO test that plotting with no sensitivity works
    def feature_0d(self,
                   feature,
                   max_legend_size=5,
                   sensitivity="first",
                   hardcopy=True,
                   show=False):

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total", None]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total or None, not {}".format(sensitivity))


        sensitivity, label = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if self.data.ndim(feature) != 0:
            raise ValueError("{} is not a 0D feature".format(feature))

        for data_type in ["mean", "variance", "percentile_5", "percentile_95"]:
            if data_type not in self.data[feature]:
                msg = "{data_type} for {feature} does not exist. Unable to plot"
                self.logger.warning(msg.format(data_type=data_type,feature=feature))
                return

        if len(self.data.uncertain_parameters) > max_legend_size:
            legend_size = max_legend_size
        else:
            legend_size = len(self.data.uncertain_parameters)

        legend_width = np.ceil(len(self.data.uncertain_parameters)/float(max_legend_size))

        width = 0.2
        distance = 0.5

        xlabels = ["mean", "variance", "$P_5$", "$P_{95}$"]
        xticks = [0, width, distance + width, distance + 2*width]

        values = [self.data[feature].mean, self.data[feature].variance,
                  self.data[feature].percentile_5, self.data[feature].percentile_95]

        ylabel = self.data.get_labels(feature)[0]

        ax = prettyBar(values,
                       index=xticks,
                       xlabels=xlabels,
                       ylabel=ylabel,
                       palette=get_colormap_tableu20())

        if sensitivity in self.data[feature]:
            pos = 2*distance + 2*width

            ax2 = ax.twinx()

            spines_color(ax2, edges={"top": "None", "bottom": "None",
                                     "right": axis_grey, "left": "None"})
            ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                            color=axis_grey, labelcolor="black", labelsize=labelsize)
            ax2.set_ylabel(label.capitalize(), fontsize=fontsize)
            ax2.set_ylim([0, 1.05])


            i = 0
            legend_bars = []
            colors = get_colormap_tableu20()

            for parameter in self.data.uncertain_parameters:

                l = ax2.bar(pos, self.data[feature][sensitivity][i], width=width,
                            align='center', color=colors[4+i], linewidth=0)

                legend_bars.append(l)

                i += 1
                pos += width

            xticks.append(pos - (i/2. + 0.5)*width)
            xlabels.append(sensitivity.split("_")[0] + " " + sensitivity.split("_")[1])

            location = (0.5, 1.01 + legend_width*0.095)
            plt.legend(legend_bars,
                       self.data.uncertain_parameters,
                       loc='upper center',
                       bbox_to_anchor=location,
                       ncol=legend_size)

            # lgd.get_frame().set_edgecolor(axis_grey)

            fig = plt.gcf()
            fig.subplots_adjust(top=(0.91 - legend_width*0.053))


        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)

        if len(self.data.uncertain_parameters) > 3:
            for tick in ax.get_xticklabels()[:2]:
                tick.set_rotation(-25)


        plt.suptitle(feature.replace("_", " "), fontsize=titlesize)

        if sensitivity is None:
            save_name = feature + self.figureformat
        else:
            save_name = feature + "_" + sensitivity + self.figureformat

        if hardcopy:
            plt.savefig(os.path.join(self.folder, save_name))

        if show:
            plt.show()
        else:
            plt.close()


        return ax


    def sensitivity_sum(self,
                          feature,
                          sensitivity="first",
                          hardcopy=True,
                          show=False):

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature not in self.data:
            raise ValueError("{} is not a feature".format(feature))

        if sensitivity + "_sum" not in self.data[feature]:
            msg = "{sensitivity}_sum of {feature} does not exist. Unable to plot {sensitivity}_sum."
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        width = 0.2

        index = np.arange(1, len(self.data.uncertain_parameters)+1)*width

        prettyBar(self.data[feature][sensitivity + "_sum"],
                  title="Normalized sum of " + title + ", " + feature.replace("_", " "),
                  xlabels=self.data.uncertain_parameters,
                  ylabel="Normalized sum of " + title,
                  nr_colors=len(self.data.uncertain_parameters),
                  index=index)


        plt.ylim([0, 1])

        save_name = feature + "_" + sensitivity + "_sum" + self.figureformat

        if hardcopy:
            plt.savefig(os.path.join(self.folder, save_name))

        if show:
            plt.show()
        else:
            plt.close()


    def sensitivity_sum_all(self,
                            sensitivity="first",
                            hardcopy=True,
                            show=False):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        if sensitivity == "first":
            sensitivity = "sobol_first"
        elif sensitivity == "total":
            sensitivity = "sobol_total"

        for feature in self.data:
            if sensitivity + "_sum" in self.data[feature]:
                self.sensitivity_sum(feature=feature,
                                     sensitivity=sensitivity,
                                     hardcopy=hardcopy,
                                     show=show)


    def features_0d(self, sensitivity="first", hardcopy=True, show=False):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        for feature in self.data:
            if self.data.ndim(feature) == 0:
                self.feature_0d(feature, sensitivity=sensitivity, hardcopy=hardcopy, show=show)



    # TODO Not Tested
    def plot_folder(self, data_dir):
        self.logger.info("Plotting all data in folder")

        for f in glob.glob(os.path.join(data_dir, "*")):
            self.load(f.split(os.path.sep)[-1])

            self.plot_all()



    # def plot_allNoSensitivity(self, sensitivity="first"):
    #     if self.data is None:
    #         raise ValueError("Datafile must be loaded.")
    #
    #
    #     self.features_1d(sensitivity=sensitivity)
    #     self.features_0d(sensitivity=sensitivity)


    def plot_all(self, sensitivity="first"):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        self.features_2d()
        self.features_1d(sensitivity=sensitivity)
        self.features_0d(sensitivity=sensitivity)

        if sensitivity is not None:
            self.sensitivity_sum_all(sensitivity=sensitivity)
            self.sensitivity_sum_grid(sensitivity=sensitivity)



    # TODO find a more descriptive name
    def plot_all_sensitivities(self):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        self.plot_all(sensitivity="first")

        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.sensitivity_1d(feature=feature, sensitivity="total")
                self.sensitivity_1d_combined(feature=feature, sensitivity="total")
                self.sensitivity_1d_grid(feature=feature, sensitivity="total")

        self.features_0d(sensitivity="total")

        self.sensitivity_sum_all(sensitivity="total")
        self.sensitivity_sum_grid(sensitivity="total")


    def plot_condensed(self, sensitivity="first"):
        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.mean_variance_1d(feature=feature)
                self.prediction_interval_1d(feature=feature)

                if sensitivity in self.data[feature]:
                    self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity)

        self.features_0d(sensitivity=sensitivity)
        self.features_2d()

        if sensitivity is not None:
            self.sensitivity_sum_grid(sensitivity=sensitivity)




    def plot(self, condensed=True, sensitivity="sobol_first"):

        if condensed:
            self.plot_condensed(sensitivity=sensitivity)
        else:
            if sensitivity is "all":
                self.plot_all_sensitivities()
            else:
                self.plot_all(sensitivity)

    # def plot_allFromExploration(self, exploration_folder):
    #     self.logger.info("Plotting all data")
    #
    #     original_data_dir = self.data_dir
    #     original_folder = self.folder
    #
    #     for folder in glob.glob(os.path.join(self.data_dir, "*")):
    #         self.data_dir = os.path.join(original_data_dir, folder.split("/")[-1])
    #         self.folder = os.path.join(original_folder,
    #                                                folder.split("/")[-1])
    #
    #         for filename in glob.glob(os.path.join(folder, "*")):
    #
    #             self.loadData(filename.split("/")[-1])
    #
    #         self.plot_all()
    #
    #     self.data_dir = original_data_dir
    #     self.folder = original_folder


    def sensitivity_sum_grid(self,
                             sensitivity="first",
                             hardcopy=True,
                             show=False,
                             **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        if sensitivity == "first":
            sensitivity = "sobol_first"
        elif sensitivity == "total":
            sensitivity = "sobol_total"


        no_sensitivity = True
        for feature in self.data:
            if sensitivity + "_sum" in self.data[feature]:
                no_sensitivity = False

        if no_sensitivity:
            msg = "All {sensitivity}_sums are missing. Unable to plot {sensitivity}_sum_grid"
            self.logger.warning(msg.format(sensitivity=sensitivity))
            return

        # get size of the grid in x and y directions
        nr_plots = len(self.data)
        grid_size = np.ceil(np.sqrt(nr_plots))
        grid_x_size = int(grid_size)
        grid_y_size = int(np.ceil(nr_plots/float(grid_x_size)))

        # plt.close("all")


        set_style("seaborn-dark")
        fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size, squeeze=False, sharex='col', sharey='row')
        set_style("seaborn-white")


        # Add a larger subplot to use to set a common xlabel and ylabel

        ax = fig.add_subplot(111, zorder=-10)
        spines_color(ax, edges={"top": "None", "bottom": "None",
                                "right": "None", "left": "None"})
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        # ax.set_xlabel(self.data.xlabel)
        ax.set_ylabel('% ' + sensitivity.split("_")[0] + " sum " + sensitivity.split("_")[1])

        width = 0.2
        index = np.arange(1, len(self.data.uncertain_parameters)+1)*width


        for i in range(0, grid_x_size*grid_y_size):
            nx = i % grid_x_size
            ny = int(np.floor(i/float(grid_x_size)))

            ax = axes[ny][nx]

            if i < nr_plots:
                if sensitivity + "_sum" not in self.data[self.data.keys()[i]]:
                    msg = " Unable to plot {sensitivity}_sum_grid. {sensitivity}_sum of {feature} does not exist."
                    self.logger.warning(msg.format(sensitivity=sensitivity,
                                                   feature=self.data.keys()[i]))
                    ax.axis("off")
                    continue

                prettyBar(self.data[self.data.keys()[i]][sensitivity + "_sum"],
                          title=self.data.keys()[i].replace("_", " "),
                          xlabels=self.data.uncertain_parameters,
                          nr_colors=len(self.data.uncertain_parameters),
                          index=index,
                          ax=ax,
                          **plot_kwargs)


                for tick in ax.get_xticklabels():
                    tick.set_rotation(-30)

                ax.set_ylim([0, 1.05])
                # ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)
                ax.tick_params(labelsize=10)
            else:
                ax.axis("off")

        title = "normalized sum " + sensitivity.replace("_", " ")
        plt.suptitle(title, fontsize=titlesize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)


        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     sensitivity + "_sum_grid" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot data")
#     parser.add_argument("-d", "--data_dir",
#                         help="Directory the data is stored in", default="data")
#     parser.add_argument("-o", "--folder",
#                         help="Folders to find compare files", default="figures")

#     args = parser.parse_args()

#     figureformat = ".png"


    # plot = PlotUncertainty(data_dir=args.data_dir,
    #                        folder=args.folder,
    #                        figureformat=figureformat)
    #
    # # plot.plot_all()
    # plot.plot_allFromExploration()

    # sortByParameters(path=folder, outputpath=folder)
