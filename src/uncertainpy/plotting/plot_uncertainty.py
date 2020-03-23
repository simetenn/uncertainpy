from __future__ import absolute_import, division, print_function, unicode_literals

import os
import itertools

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from .prettyplot import prettyPlot, prettyBar
from .prettyplot import spines_color, get_current_colormap
from .prettyplot import set_style, get_colormap, reset_style
from .prettyplot import axis_grey, labelsize, fontsize, titlesize, linewidth

from ..data import Data
from ..utils.logger import setup_module_logger, get_logger


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
        The name of the data file. If given the file is loaded. If None, no file
        is loaded. Default is None.
    folder : str, optional
        The folder where to save the plots. Creates a new folder if it does not
        exist. Default is "figures/".
    figureformat : str, optional
        The format to save the plots in. Given as ".xxx". All formats supported
        by Matplotlib are available. Default is ".png",
    logger_level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging to file is performed
        Default logger level is "info".

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
    """
    def __init__(self,
                 filename=None,
                 folder="figures/",
                 figureformat=".png",
                 logger_level="info",
                 uncertain_names=[]):

        self._folder = None

        self.folder = folder
        self.figureformat = figureformat

        self.features_in_combined_plot = 3

        self.data = None
        self.uncertain_names = uncertain_names

        self._logger_level = logger_level

        logger = get_logger(self)
        if filename is not None:
            self.load(filename)
            if not self.uncertain_names:
                logger.warning("no uncertain names passed for labels - will use parameter names")
                self.uncertain_names = [i for i in self.data.uncertain_parameters]
            assert (len(self.uncertain_names) == len(self.data.uncertain_parameters)), print("You must provide a name for each parameter")

        setup_module_logger(class_instance=self, level=logger_level)

        logger.info("uncertain names are: {}".format(" ".join(self.uncertain_names)))

    def load(self, filename):
        """
        Load data from a HDF5 or Exdir file with name `filename`.

        Parameters
        ----------
        filename : str
            Name of the file to load data from.
        """
        logger = get_logger(self)
        self.data = Data(filename,
                         logger_level=self._logger_level)
        if not self.uncertain_names:
            logger.warning("no uncertain names passed for labels - will use parameter names")
            self.uncertain_names = [i for i in self.data.uncertain_parameters]
        assert (len(self.uncertain_names) == len(self.data.uncertain_parameters)), print("You must provide a name for each parameter")

    def set_data(self, data):
        """
        Set data from dictionary.

        Parameters
        ----------
        data : dict
            Data dictionary
        """
        logger = get_logger(self)
        self.data = data
        if not self.uncertain_names:
            logger.warning("no uncertain names passed for labels - will use parameter names")
            self.uncertain_names = [i for i in self.data.uncertain_parameters]
        assert (len(self.uncertain_names) == len(self.data.uncertain_parameters)), print("You must provide a name for each parameter")

    @property
    def folder(self):
        """
        The folder where to save all plots.

        Parameters
        ----------
        new_folder : str
            Name of new folder where to save all plots. The folder is created
            if it does not exist.
        """
        return self._folder

    @folder.setter
    def folder(self, new_folder):
        self._folder = new_folder

        if new_folder is not None and not os.path.isdir(new_folder):
            os.makedirs(new_folder)

    def all_evaluations(self, foldername="evaluations"):
        """
        Plot all evaluations for all model and features.

        Parameters
        ----------
        foldername : str, optional
            Name of folder where to save all plots. The folder is created
            if it does not exist. Default folder is named "evaluations".
        """
        for feature in self.data.data:
            self.evaluations(feature=feature, foldername=foldername)

    def evaluations(self, feature=None, foldername="", **plot_kwargs):
        """
        Plot all evaluations for a specific model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        foldername : str, optional
            Name of folder where to save all plots. The folder is created
            if it does not exist. Default folder is named "featurename_evaluations".
        xscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for x axis.
        yscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for y axis.
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        NotImplementedError
            If the model/feature have more than 2 dimensions.
        AttributeError
            If the dimensions of the evaluations is not valid.
        """
        logger = get_logger(self)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        save_folder = os.path.join(self.folder, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        dimension = self.data.ndim(feature)
        if dimension is None:
            logger.warning("No evaluations to plot")
        elif dimension == 0:
            self.evaluations_0d(feature=feature, foldername=foldername, **plot_kwargs)

        elif dimension == 1:
            self.evaluations_1d(feature=feature, foldername=foldername, **plot_kwargs)

        elif dimension == 2:
            self.evaluations_2d(feature=feature, foldername=foldername, **plot_kwargs)

        elif dimension > 2:
            raise NotImplementedError(">2 dimensional plots not implemented.")
        else:
            raise AttributeError("Dimension of evaluations is not valid: dim {}".format(dimension))

    def evaluations_0d(self, feature=None, foldername="", **plot_kwargs):
        """
        Plot all 0D evaluations for a specific model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        foldername : str, optional
            Name of folder where to save all plots. The folder is created
            if it does not exist.Default folder is named "featurename_evaluations".
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the evaluations are not 0 dimensional.
        """
        logger = get_logger(self)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data or "evaluations" not in self.data[feature]:
            logger.warning("No {} evaluations to plot.".format(feature))
            return

        if self.data.ndim(feature) != 0:
            raise ValueError("{} is not a 0 dimensional feature".format(feature))

        save_folder = os.path.join(self.folder, foldername, feature + "_evaluations")
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        prettyPlot(self.data[feature].evaluations,
                   xlabel=r"Evaluation #number",
                   ylabel=self.data.get_labels(feature)[0],
                   title="{}, evaluations".format(feature.replace("_", " ")),
                   new_figure=True,
                   palette="husl",
                   **plot_kwargs)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "evaluations" + self.figureformat))
        plt.close()

        reset_style()

    def evaluations_1d(self, feature=None, foldername="", xscale='linear', yscale='linear', **plot_kwargs):
        """
        Plot all 1D evaluations for a specific model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        foldername : str, optional
            Name of folder where to save all plots. The folder is created
            if it does not exist. Default folder is named "featurename_evaluations".
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the evaluations are not 1 dimensional.
        """
        logger = get_logger(self)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data or "evaluations" not in self.data[feature]:
            logger.warning("No model evaluations to plot.")
            return

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1 dimensional feature".format(feature))

        save_folder = os.path.join(self.folder, foldername, feature + "_evaluations")
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        if not self.data.model_ignore:
            if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
                time = np.arange(0, len(self.data[feature].evaluations[0]))
            else:
                time = self.data[feature].time

        padding = len(str(len(self.data[feature].evaluations) + 1))
        for i, evaluation in enumerate(self.data[feature].evaluations):

            if self.data.model_ignore:
                if self.data[feature].time[i] is None or np.all(np.isnan(self.data[feature].time[i])):
                    time = np.arange(0, len(self.data[feature].evaluations[i]))
                else:
                    time = self.data[feature].time[i]

            ax = prettyPlot(time, evaluation,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            title="{}, evaluation {:d}".format(feature.replace("_", " "), i),
                            new_figure=True,
                            palette="husl",
                            **plot_kwargs)
            ax.set_xlim([min(time), max(time)])
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder,
                                     "evaluation_{0:0{1}d}".format(i, padding) + self.figureformat))
            plt.close()

        reset_style()

    def evaluations_2d(self, feature=None, foldername="", **plot_kwargs):
        """
        Plot all 2D evaluations for a specific model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        foldername : str, optional
            Name of folder where to save all plots. The folder is created
            if it does not exist. Default folder is named
            "featurename_evaluations".
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the evaluations are not 2 dimensional.
        """
        logger = get_logger(self)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data or "evaluations" not in self.data[feature]:
            logger.warning("No model evaluations to plot.")
            return

        if self.data.ndim(feature) != 2:
            raise ValueError("{} is not a 2 dimensional feature.".format(feature))

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

        reset_style()

    def attribute_feature_1d(self,
                             feature=None,
                             attribute="mean",
                             attribute_name="mean",
                             hardcopy=True,
                             show=False,
                             xscale="linear",
                             yscale="linear",
                             title=None,
                             **plot_kwargs):
        """
        Plot a 1 dimensional attribute for a specific model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        attribute : {"mean", "variance"}, optional
            Attribute to plot, either the mean or variance. Default is "mean".
        attribute_name : str
            Name of the attribute, used as title and name of the plot.
            Default is "mean".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        xscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for x axis.
        yscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for y axis.
        title: string, optional
            Choose a customized title
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        ValueError
            If the attribute is not a supported attribute, either "mean" or
            "variance".
        """
        logger = get_logger(self)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1 dimensional feature".format(feature))

        if attribute not in ["mean", "variance"]:
            raise ValueError("{} is not a supported attribute".format(attribute))

        if attribute not in self.data[feature]:
            msg = " Unable to plot {attribute_name}. {attribute_name} of {feature} does not exist."
            logger.warning(msg.format(attribute_name=attribute, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][attribute]))
        else:
            time = self.data[feature].time

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        if title is None:
            title = feature + ", " + attribute_name
        ax = prettyPlot(time, self.data[feature][attribute],
                        title.replace("_", " "), xlabel, ylabel,
                        nr_colors=3,
                        palette="husl",
                        **plot_kwargs)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

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

        reset_style()

    def attribute_feature_2d(self,
                             feature=None,
                             attribute="mean",
                             attribute_name="mean",
                             hardcopy=True,
                             show=False,
                             **plot_kwargs):
        """
        Plot a 2 dimensional attribute for a specific model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        attribute : {"mean", "variance"}, optional
            Attribute to plot, either the mean or variance. Default is "mean".
        attribute_name : str
            Name of the attribute, used as title and name of the plot.
            Default is "mean".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 2 dimensional.
        ValueError
            If the attribute is not a supported attribute, either "mean" or
            "variance".
        """
        logger = get_logger(self)

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
            logger.warning(msg.format(attribute_name=attribute, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            extent = None
        else:
            extent = [self.data[feature].time[0], self.data[feature].time[-1],
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

        reset_style()

    def mean_1d(self, feature, hardcopy=True, show=False, xscale='linear', yscale='linear', title=None, **plot_kwargs):
        """
        Plot the mean for a specific 1 dimensional model/feature.

        Parameters
        ----------
        feature : str
            The name of the model/feature.
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        xscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for x axis.
        yscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for y axis.
        title: string, optional
            Choose a customized title
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        """
        self.attribute_feature_1d(feature,
                                  attribute="mean",
                                  attribute_name="mean",
                                  hardcopy=hardcopy,
                                  show=show,
                                  xscale=xscale,
                                  yscale=yscale,
                                  title=title,
                                  color=0,
                                  **plot_kwargs)

    def variance_1d(self, feature, hardcopy=True, show=False, xscale='linear', yscale='linear', title=None, **plot_kwargs):
        """
        Plot the variance for a specific 1 dimensional model/feature.

        Parameters
        ----------
        feature : str
            The name of the model/feature.
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        xscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for x axis.
        yscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for y axis.
        title: string, optional
            Choose a customized title
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        """
        self.attribute_feature_1d(feature,
                                  attribute="variance",
                                  attribute_name="variance",
                                  hardcopy=hardcopy,
                                  show=show,
                                  xscale=xscale,
                                  yscale=yscale,
                                  title=title,
                                  color=2,
                                  **plot_kwargs)

    def mean_2d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        """
        Plot the mean for a specific 2 dimensional model/feature.

        Parameters
        ----------
        feature : str
            The name of the model/feature.
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 2 dimensional.
        """
        self.attribute_feature_2d(feature,
                                  attribute="mean",
                                  attribute_name="mean",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)

    def variance_2d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        """
        Plot the variance for a specific 2 dimensional model/feature.

        Parameters
        ----------
        feature : str
            The name of the model/feature.
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 2 dimensional.
        """
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
                         xscale='linear',
                         yscale='linear',
                         title=None,
                         **plot_kwargs):
        """
        Plot the mean and variance for a specific 1 dimensional model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        xscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for x axis.
        yscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for y axis.
        title: string, optional
            Choose a customized title
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        """
        logger = get_logger(self)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if "mean" not in self.data[feature] or "variance" not in self.data[feature]:
            msg = "Mean and/or variance of {feature} does not exist. ".format(feature=feature) \
                  + "Unable to plot mean and variance"
            logger.warning(msg)
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature].mean))
        else:
            time = self.data[feature].time

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        style = "seaborn-white"
        if title is None:
            title = feature + ", mean and variance"
        ax = prettyPlot(time, self.data[feature].mean,
                        title.replace("_", " "), xlabel, ylabel + ", mean",
                        style=style,
                        nr_colors=3,
                        palette="husl",
                        **plot_kwargs)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        colors = get_current_colormap()

        ax2 = ax.twinx()
        color = 0
        color_2 = 2

        spines_color(ax2, edges={"top": "None", "bottom": "None",
                                 "right": colors[color_2], "left": "None"})
        ax2.tick_params(axis="y", which="both", right=False, left=False, labelright=True,
                        color=colors[color_2], labelcolor=colors[color_2], labelsize=labelsize)
        ax2.set_ylabel("(" + ylabel + r")$^2$, variance", color=colors[color_2], fontsize=labelsize)

        # ax2.set_ylim([min(self.data.variance[feature]), max(self.data.variance[feature])])

        ax2.plot(time, self.data[feature].variance,
                 color=colors[color_2], linewidth=linewidth, antialiased=True)

        ax2.yaxis.offsetText.set_fontsize(fontsize)
        ax2.yaxis.offsetText.set_color(colors[color_2])

        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_edgecolor(colors[color_2])

        ax.tick_params(axis="y", color=colors[color], labelcolor=colors[color])
        ax.spines["left"].set_edgecolor(colors[color])
        ax.set_ylabel(ylabel + ", mean", color=colors[color], fontsize=labelsize)

        ax2.set_xlim([min(time), max(time)])
        ax.set_xlim([min(time), max(time)])
        ax2.set_yscale(yscale)
        ax2.set_xscale(xscale)

        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_mean-variance" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()

        reset_style()
        #
        # if not show or not hardcopy:
        #     return ax, ax2

    def prediction_interval_1d(self,
                               feature=None,
                               hardcopy=True,
                               show=False,
                               title=None,
                               xscale='linear',
                               yscale='linear',
                               **plot_kwargs):
        """
        Plot the prediction interval for a specific 1 dimensional model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        title: string, optional
            Choose a customized title
        xscale: string, optional
            Choose the axis scale for the xaxis.
        yscale: string, optional
            Choose the axis scale for the yaxis.
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        """
        logger = get_logger(self)

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
            logger.warning(msg.format(feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature].mean))
        else:
            time = self.data[feature].time

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        if title is None:
            title = feature.replace("_", " ") + ", 90% prediction interval"
        ax = prettyPlot(time, self.data[feature].mean, title=title,
                        xlabel=xlabel, ylabel=ylabel,
                        color=0,
                        nr_colors=3,
                        palette="husl",
                        **plot_kwargs)

        colors = get_current_colormap()
        ax.fill_between(time,
                        self.data[feature].percentile_5,
                        self.data[feature].percentile_95,
                        alpha=0.5, color=colors[0],
                        linewidth=0)

        ax.set_xlim([min(time), max(time)])
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        plt.legend(["Mean", "90% prediction interval"], loc="best")

        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_prediction-interval" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()

        reset_style()

    def sensitivity_1d(self,
                       feature=None,
                       sensitivity="first",
                       hardcopy=True,
                       show=False,
                       xscale='linear',
                       yscale='linear',
                       title=None,
                       **plot_kwargs):
        """
        Plot the sensitivity for a specific 1 dimensional model/feature. The
        Sensitivity for each parameter is plotted in sepearate figures.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. Default is "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        xscale: string, optional
            Choose the axis scale for the xaxis.
        yscale: string, optional
            Choose the axis scale for the yaxis.
        title: string, optional
            Choose a customized title
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            or "total".
        """
        logger = get_logger(self)

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title_tmp = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            time = self.data[feature].time

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        for i in range(len(self.data[feature][sensitivity])):
            if title is None:
                title = title_tmp + ", " + feature.replace("_", " ") + " - " + self.uncertain_names[i]
            ax = prettyPlot(time, self.data[feature][sensitivity][i],
                            title=title,
                            xlabel=xlabel,
                            ylabel=title,
                            color=i,
                            palette="husl",
                            nr_colors=len(self.data.uncertain_parameters), **plot_kwargs)
            # plt.ylim([0, 1.05])
            ax.set_xlim([min(time), max(time)])
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            plt.tight_layout()

            if hardcopy:
                plt.savefig(os.path.join(self.folder,
                                         feature + "_" + sensitivity + "_"
                                         + self.data.uncertain_parameters[i] + self.figureformat))

            if show:
                plt.show()
            else:
                plt.close()

        reset_style()

    def sensitivity_1d_grid(self,
                            feature=None,
                            sensitivity="first",
                            hardcopy=True,
                            show=False,
                            xscale="linear",
                            yscale="linear",
                            title=None,
                            **plot_kwargs):
        """
        Plot the sensitivity for a specific 1 dimensional model/feature. The
        Sensitivity for each parameter is plotted in the same figure, but
        separate plots.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. Default is "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        xscale: string, optional
            Choose the axis scale for the xaxis.
        yscale: string, optional
            Choose the axis scale for the yaxis.
        title: string, optional
            Choose a customized title
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            or "total".
        """
        logger = get_logger(self)

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title_tmp = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            time = self.data[feature].time

        parameter_names = self.uncertain_names

        # get size of the grid in x and y directions
        nr_plots = len(parameter_names)
        grid_size = np.ceil(np.sqrt(nr_plots))
        grid_x_size = int(grid_size)
        grid_y_size = int(np.ceil(nr_plots / float(grid_x_size)))

        set_style("seaborn-darkgrid")
        fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size, squeeze=False, sharex="col", sharey="row")

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        if title is None:
            title = title_tmp

        # Add a larger subplot to use to set a common xlabel and ylabel
        set_style("seaborn-white")
        ax = fig.add_subplot(111, zorder=-10)
        spines_color(ax, edges={"top": "None", "bottom": "None",
                                "right": "None", "left": "None"})
        ax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(title)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        for i in range(0, grid_x_size * grid_y_size):
            nx = i % grid_x_size
            ny = int(np.floor(i / float(grid_x_size)))

            ax = axes[ny][nx]

            if i < nr_plots:
                prettyPlot(time, self.data[feature][sensitivity][i],
                           title=parameter_names[i],
                           color=i,
                           nr_colors=nr_plots,
                           ax=ax,
                           palette="husl",
                           **plot_kwargs)

                # for tick in ax.get_xticklabels():
                #     tick.set_rotation(-30)

                ax.set_ylim([0, 1.05])
                ax.set_xlim([min(time), max(time)])
                # ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)
                ax.tick_params(labelsize=10)
            else:
                ax.set_axis_off()

        title = title + ", " + feature.replace("_", " ")
        plt.suptitle(title, fontsize=titlesize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_" + sensitivity + "_grid" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()

        reset_style()

    def sensitivity_1d_combined(self,
                                feature=None,
                                sensitivity="first",
                                hardcopy=True,
                                show=False,
                                xscale="linear",
                                yscale="linear",
                                title=None,
                                **plot_kwargs):
        """
        Plot the sensitivity for a specific 1 dimensional model/feature. The
        Sensitivity for each parameter is plotted in the same plot.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. Default is "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        xscale: string, optional
            Choose the axis scale for the xaxis.
        yscale: string, optional
            Choose the axis scale for the yaxis.
        title: string, optional
            Choose a customized title
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            or "total".
        """
        logger = get_logger(self)

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title_tmp = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if self.data.ndim(feature) != 1:
            raise ValueError("{} is not a 1D feature".format(feature))

        if sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            time = self.data[feature].time

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        for i in range(len(self.data[feature][sensitivity])):
            if title is None:
                title = title_tmp + ", " + feature.replace("_", " ")
            prettyPlot(time,
                       self.data[feature][sensitivity][i],
                       title=title,
                       xlabel=xlabel,
                       ylabel=title_tmp,
                       new_figure=False,
                       color=i,
                       palette="husl",
                       nr_colors=len(self.data.uncertain_parameters),
                       label=self.uncertain_names[i],
                       **plot_kwargs)

        plt.ylim([0, 1.05])
        plt.xlim([min(time), max(time)])
        plt.xscale(xscale)
        plt.yscale(yscale)
        if len(self.data[feature][sensitivity]) > 4:
            plt.xlim([time[0], 1.3 * time[-1]])

        plt.legend()
        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_" + sensitivity + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()

        reset_style()

    def features_1d(self, sensitivity="first", xscale='linear', yscale='linear', title=None):
        """
        Plot all data for all 1 dimensional model/features.

        For each model/feature plots ``mean_1d``, ``variance_1d``,
        ``mean_variance_1d``, and ``prediction_interval_1d``. If sensitivity
        also plot ``sensitivity_1d``, ``sensitivity_1d_combined``, and
        ``sensitivity_1d_grid``.

        Parameters
        ----------
        sensitivity : {"sobol_first", "first", "sobol_total", "total", None}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. If None, no sensitivity is plotted. Default is
            "first".
        xscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for x axis.
        yscale: {"linear", "log", "symlog", "logit", ...}, optional
            Choose scale for y axis.
        title: string, optional
            Choose a customized title

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            "total" or None.

        See also
        --------
        uncertainpy.plotting.PlotUncertainty.mean_1d
        uncertainpy.plotting.PlotUncertainty.variance_1d
        uncertainpy.plotting.PlotUncertainty.mean_variance_1d
        uncertainpy.plotting.PlotUncertainty.prediction_interval_1d
        uncertainpy.plotting.PlotUncertainty.sensitivity_1d
        uncertainpy.plotting.PlotUncertainty.sensitivity_1d_combined
        uncertainpy.plotting.PlotUncertainty.sensitivity_1d_grid
        """
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total", None]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total or None, not {}".format(sensitivity))

        sensitivity, label = self.convert_sensitivity(sensitivity)

        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.mean_1d(feature=feature, xscale=xscale, yscale=yscale, title=title)
                self.variance_1d(feature=feature, xscale=xscale, yscale=yscale, title=title)
                self.mean_variance_1d(feature=feature, xscale=xscale, yscale=yscale, title=title)
                self.prediction_interval_1d(feature=feature, xscale=xscale, yscale=yscale, title=title)

                if sensitivity in self.data[feature]:
                    self.sensitivity_1d(feature=feature, sensitivity=sensitivity, xscale=xscale, yscale=yscale, title=title)
                    self.sensitivity_1d_combined(feature=feature, sensitivity=sensitivity, xscale=xscale, yscale=yscale, title=title)
                    self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity, xscale=xscale, yscale=yscale, title=title)

    def convert_sensitivity(self, sensitivity):
        """
        Convert a sensitivity str to the correct sensitivity attribute, and a
        full name.

        Parameters
        ----------
        sensitivity : {"sobol_first", "first", "sobol_total", "total", None}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices.

        Returns
        -------
        sensitivity : str
            Name of the sensitivity attribute. Either sobol_first",
            "sobol_total", or the unchanged input.
        full_text : str
            Complete name of the sensitivity. Either "", or
            "first order Sobol indices" or "total order Sobol indices".
        """
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
        """
        Plot all implemented plots for all 2 dimensional model/features.
        For each model/feature plots ``mean_2d``, and ``variance_2d``.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        """
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
                   sensitivity="first",
                   hardcopy=True,
                   show=False,
                   max_legend_size=5):
        """
        Plot all attributes (mean, variance, p_05, p_95 and sensitivity of it
        exists) for a 0 dimensional model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        sensitivity : {"sobol_first", "first", "sobol_total", "total", None}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. If None, no sensitivity is plotted. Default is
            "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        max_legend_size : int, optional
            The max number of legends in a row. Default is 5.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 0 dimensional.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            "total" or None.
        """
        logger = get_logger(self)

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total", None]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total or None, not {}".format(sensitivity))

        sensitivity, label = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if self.data.ndim(feature) != 0:
            raise ValueError("{} is not a 0D feature".format(feature))

        for data_type in ["mean", "variance", "percentile_5", "percentile_95"]:
            if data_type not in self.data[feature]:
                msg = "{data_type} for {feature} does not exist. Unable to plot."
                logger.warning(msg.format(data_type=data_type, feature=feature))
                return

        if len(self.data.uncertain_parameters) > max_legend_size:
            legend_size = max_legend_size
        else:
            legend_size = len(self.data.uncertain_parameters)

        legend_width = np.ceil(len(self.data.uncertain_parameters) / float(max_legend_size))

        width = 0.2
        distance = 0.5

        xlabels = ["Mean", "Variance", "$P_5$", "$P_{95}$"]
        xticks = [0, width, distance + width, distance + 2 * width]

        values = [self.data[feature].mean, self.data[feature].variance,
                  self.data[feature].percentile_5, self.data[feature].percentile_95]

        ylabel = self.data.get_labels(feature)[0]

        ax = prettyBar(values,
                       index=xticks,
                       xlabels=xlabels,
                       ylabel=ylabel,
                       palette="Paired",
                       style="seaborn-white")

        if sensitivity in self.data[feature]:
            pos = 2 * distance + 2 * width

            ax2 = ax.twinx()

            spines_color(ax2, edges={"top": "None", "bottom": "None",
                                     "right": axis_grey, "left": "None"})
            ax2.tick_params(axis="y", which="both", right=True, left=False, labelright=True,
                            color=axis_grey, labelcolor="black", labelsize=labelsize)
            ax2.set_ylabel(label, fontsize=labelsize)
            ax2.set_ylim([0, 1.05])

            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_edgecolor(axis_grey)

            i = 0
            legend_bars = []
            colors = get_colormap(palette="husl", nr_colors=len(self.data.uncertain_parameters))

            for parameter in self.data.uncertain_parameters:
                ll = ax2.bar(pos, self.data[feature][sensitivity][i], width=width,
                             align="center", color=colors[i], linewidth=0)
                legend_bars.append(ll)

                i += 1
                pos += width

            xticks.append(pos - (i / 2. + 0.5) * width)
            xlabels.append(sensitivity.split("_")[0] + " " + sensitivity.split("_")[1])

            location = (0.5, 1.01 + legend_width * 0.095)
            plt.legend(legend_bars,
                       self.uncertain_names,
                       loc="upper center",
                       bbox_to_anchor=location,
                       ncol=legend_size)

            # lgd.get_frame().set_edgecolor(axis_grey)

            fig = plt.gcf()
            fig.subplots_adjust(top=(0.91 - legend_width * 0.053))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)

        if len(self.data.uncertain_parameters) > 3:
            for tick in ax.get_xticklabels()[:2]:
                tick.set_rotation(-25)

        plt.suptitle(feature.replace("_", " "), fontsize=titlesize)

        if sensitivity is None or sensitivity not in self.data[feature]:
            plt.subplots_adjust(top=0.93)

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

        reset_style()

        # return ax

    def average_sensitivity(self,
                            feature=None,
                            sensitivity="first",
                            hardcopy=True,
                            title=None,
                            show=False):
        """
        Plot the average of the sensitivity for a specific model/feature.

        Parameters
        ----------
        feature : {None, str}
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. Default is "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        title: string, optional
            Choose a customized title
        show : bool, optional
            If the plot should be shown on screen. Default is False.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            or "total".
        ValueError
            If feature does not exist.
        """
        logger = get_logger(self)

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title_tmp = self.convert_sensitivity(sensitivity)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data:
            raise ValueError("{} is not a feature".format(feature))

        if sensitivity + "_average" not in self.data[feature]:
            msg = "{sensitivity}_average of {feature} does not exist. Unable to plot {sensitivity}_average."
            logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        width = 0.2

        index = np.arange(1, len(self.data.uncertain_parameters) + 1) * width

        if title is None:
            title = "Average of " + title_tmp + ", " + feature.replace("_", " ")
        prettyBar(self.data[feature][sensitivity + "_average"],
                  title=title,
                  xlabels=self.uncertain_names,
                  ylabel="Average of " + title_tmp,
                  nr_colors=len(self.data.uncertain_parameters),
                  palette="husl",
                  index=index,
                  style="seaborn-darkgrid")

        plt.ylim([0, 1])

        save_name = feature + "_" + sensitivity + "_average" + self.figureformat

        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder, save_name))

        if show:
            plt.show()
        else:
            plt.close()

        reset_style()

    def average_sensitivity_all(self,
                                sensitivity="first",
                                hardcopy=True,
                                title=None,
                                show=False):
        """
        Plot the average of the sensitivity for all model/features.

        Parameters
        ----------
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. Default is "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        title: string, optional
            Choose a customized title
        show : bool, optional
            If the plot should be shown on screen. Default is False.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            or "total".
        """
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, _ = self.convert_sensitivity(sensitivity)

        for feature in self.data:
            if sensitivity + "_average" in self.data[feature]:
                self.average_sensitivity(feature=feature,
                                         sensitivity=sensitivity,
                                         hardcopy=hardcopy,
                                         title=title,
                                         show=show)

    def features_0d(self, sensitivity="first", hardcopy=True, show=False):
        """
        Plot the results for all 0 dimensional model/features.

        Parameters
        ----------
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. Default is "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            or "total".
        """
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        for feature in self.data:
            if self.data.ndim(feature) == 0:
                self.feature_0d(feature, sensitivity=sensitivity, hardcopy=hardcopy, show=show)

    # # TODO Not Tested
    # def plot_folder(self, data_dir):
    #     self.logger.info("Plotting all data in folder")

    #     for f in glob.glob(os.path.join(data_dir, "*")):
    #         self.load(f.split(os.path.sep)[-1])

    #         self.plot_all()

    # def plot_allNoSensitivity(self, sensitivity="first"):
    #     if self.data is None:
    #         raise ValueError("Datafile must be loaded.")
    #
    #
    #     self.features_1d(sensitivity=sensitivity)
    #     self.features_0d(sensitivity=sensitivity)

    def plot_all(self, sensitivity="first"):
        """
        Plot the results for all model/features, with the chosen sensitivity.

        Parameters
        ----------
        sensitivity : {"sobol_first", "first", "sobol_total", "total", None}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. If None, no sensitivity is plotted.
            Default is "first".

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            "total", or None.
        """
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        self.features_2d()
        self.features_1d(sensitivity=sensitivity)
        self.features_0d(sensitivity=sensitivity)

        if sensitivity is not None:
            self.average_sensitivity_all(sensitivity=sensitivity)
            self.average_sensitivity_grid(sensitivity=sensitivity)

    # TODO find a more descriptive name
    def plot_all_sensitivities(self):
        """
        Plot the results for all model/features, with all sensitivities.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        """
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        self.plot_all(sensitivity="first")

        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.sensitivity_1d(feature=feature, sensitivity="total")
                self.sensitivity_1d_combined(feature=feature, sensitivity="total")
                self.sensitivity_1d_grid(feature=feature, sensitivity="total")

        self.features_0d(sensitivity="total")

        self.average_sensitivity_all(sensitivity="total")
        self.average_sensitivity_grid(sensitivity="total")

    def plot_condensed(self, sensitivity="first"):
        """
        Plot the subset of data that shows all information in the most concise
        way, with the chosen sensitivity.

        Parameters
        ----------
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. If None, no sensitivity is plotted.
            Default is "first".

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            "total", or None.
        """
        if sensitivity not in ["sobol_first", "first", "sobol_total", "total", None]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, _ = self.convert_sensitivity(sensitivity)

        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.mean_variance_1d(feature=feature)
                self.prediction_interval_1d(feature=feature)

                if sensitivity in self.data[feature]:
                    self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity)

        self.features_0d(sensitivity=sensitivity)
        self.features_2d()

        if sensitivity is not None:
            self.average_sensitivity_grid(sensitivity=sensitivity)

    def plot(self, condensed=True, sensitivity="first"):
        """
        Plot the subset of data that shows all information in the most concise
        way, with the chosen sensitivity.

        Parameters
        ----------
        condensed : bool, optional
            If the results should be plotted in the most concise way. If not, all
            plots are created. Default is True.
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. If None, no sensitivity is plotted.
            Default is "first".

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            "total", or None.
        """
        if condensed:
            self.plot_condensed(sensitivity=sensitivity)
        else:
            if sensitivity is "all":
                self.plot_all_sensitivities()
            else:
                self.plot_all(sensitivity)

    def average_sensitivity_grid(self,
                                 sensitivity="first",
                                 hardcopy=True,
                                 title=None,
                                 show=False,
                                 **plot_kwargs):
        """
        Plot the average of the sensitivity for all model/features in
        their own plots in the same figure.

        Parameters
        ----------
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            Which Sobol indices to plot. "sobol_first" and "first" is the first
            order Sobol indices, while "sobol_total" and "total" are the total
            order Sobol indices. Default is "first".
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If sensitivity is not one of "sobol_first", "first", "sobol_total",
            or "total".
        """
        logger = get_logger(self)

        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title_tmp = self.convert_sensitivity(sensitivity)

        no_sensitivity = True
        for feature in self.data:
            if sensitivity + "_average" in self.data[feature]:
                no_sensitivity = False

        if no_sensitivity:
            msg = "All {sensitivity}_averages are missing. Unable to plot {sensitivity}_average_grid"
            logger.warning(msg.format(sensitivity=sensitivity))
            return

        # get size of the grid in x and y directions
        nr_plots = len(self.data)
        grid_size = np.ceil(np.sqrt(nr_plots))
        grid_x_size = int(grid_size)
        grid_y_size = int(np.ceil(nr_plots / float(grid_x_size)))

        # plt.close("all")

        set_style("seaborn-dark")
        fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size, squeeze=False, sharex="col", sharey="row")
        set_style("seaborn-white")

        # Add a larger subplot to use to set a common xlabel and ylabel

        ax = fig.add_subplot(111, zorder=-10)
        spines_color(ax, edges={"top": "None", "bottom": "None",
                                "right": "None", "left": "None"})
        ax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Average of " + title_tmp)

        width = 0.2
        index = np.arange(1, len(self.data.uncertain_parameters) + 1) * width

        features = list(self.data.keys())
        for i in range(0, grid_x_size * grid_y_size):
            nx = i % grid_x_size
            ny = int(np.floor(i / float(grid_x_size)))

            ax = axes[ny][nx]

            if i < nr_plots:
                if sensitivity + "_average" not in self.data[features[i]]:
                    msg = " Unable to plot {sensitivity}_average_grid. {sensitivity}_average of {feature} does not exist."
                    logger.warning(msg.format(sensitivity=sensitivity,
                                              feature=features[i]))
                    ax.set_axis_off()
                    continue

                prettyBar(self.data[features[i]][sensitivity + "_average"],
                          title=features[i].replace("_", " "),
                          xlabels=self.uncertain_names,
                          nr_colors=len(self.data.uncertain_parameters),
                          index=index,
                          palette="husl",
                          ax=ax,
                          **plot_kwargs)

                for tick in ax.get_xticklabels():
                    tick.set_rotation(-30)

                ax.set_ylim([0, 1.05])
                # ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)
                ax.tick_params(labelsize=fontsize)
            else:
                ax.set_axis_off()

        if title is None:
            title = "Average of " + title_tmp
        plt.suptitle(title, fontsize=titlesize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     sensitivity + "_average_grid" + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()

        reset_style()

    def prediction_interval_sensitivity_1d(self,
                                           feature=None,
                                           hardcopy=True,
                                           show=False,
                                           sensitivity="first",
                                           title=None,
                                           xscale='linear',
                                           yscale='linear',
                                           use_markers=False,
                                           legend_locs=None,
                                           **plot_kwargs):
        """
        Plot the prediction interval for a specific 1 dimensional model/feature.

        Parameters
        ----------
        feature : {None, str}, optional
            The name of the model/feature. If None, the name of the model is
            used. Default is None.
        hardcopy : bool, optional
            If the plot should be saved to file. Default is True.
        show : bool, optional
            If the plot should be shown on screen. Default is False.
        title: string, optional
            Choose a customized title
        xscale: string, optional
            Choose the axis scale for the xaxis.
        yscale: string, optional
            Choose the axis scale for the yaxis.
        use_markers: bool, optional
            Use markers for the sensitivity data.
        **plot_kwargs, optional
            Matplotlib plotting arguments.

        Raises
        ------
        ValueError
            If a Datafile is not loaded.
        ValueError
            If the model/feature is not 1 dimensional.
        """
        logger = get_logger(self)

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
            logger.warning(msg.format(feature=feature))
            return

        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        sensitivity, title_tmp = self.convert_sensitivity(sensitivity)

        if sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if self.data[feature].time is None or np.all(np.isnan(self.data[feature].time)):
            time = np.arange(0, len(self.data[feature].mean))
        else:
            time = self.data[feature].time

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        # plot predicition interval
        if title is None:
            title = feature.replace("_", " ") + ", 90% prediction interval and sensitivities"
        ax = prettyPlot(time, self.data[feature].mean, title=title,
                        xlabel=xlabel, ylabel=ylabel,
                        color=0,
                        nr_colors=3,
                        palette="husl",
                        **plot_kwargs)

        colors = get_current_colormap()
        ax.fill_between(time,
                        self.data[feature].percentile_5,
                        self.data[feature].percentile_95,
                        alpha=0.5, color=colors[0],
                        linewidth=0)

        ax.set_xlim([min(time), max(time)])
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # plot sensitivity
        ax2 = ax.twinx()
        colors = sns.color_palette("husl", n_colors=len(self.data[feature][sensitivity]) + 2)
        color = 0
        color_2 = 2

        spines_color(ax2, edges={"top": "None", "bottom": "None",
                                 "right": colors[color_2], "left": "None"})
        ax2.grid(False)
        ax2.tick_params(axis="y", which="both", right=True, left=False, labelright=True,
                        color=colors[color_2], labelcolor=colors[color_2], labelsize=labelsize)
        ax2.set_ylabel(title_tmp, color=colors[color_2], fontsize=labelsize)

        # use markers for secondary axis
        marker_list = itertools.cycle(('+', 'o', '*', 'v', '<', '>', 's', 'x')) 

        for i in range(len(self.data[feature][sensitivity])):
            if use_markers is False:
                marker = None
            else:
                marker = next(marker_list)

            ax2.plot(time, self.data[feature][sensitivity][i], color=colors[i + 2],
                     linewidth=linewidth, antialiased=True, label=self.uncertain_names[i],
                     marker=marker, **plot_kwargs)

        if legend_locs is not None:
            assert (len(legend_locs) == 2, "you must provide two legend locations")
            loc1 = legend_locs[0]
            loc2 = legend_locs[1]
        else:
            loc1 = "best"
            loc2 = "lower right"

        ax2.legend(loc=loc2)
        ax2.yaxis.offsetText.set_fontsize(fontsize)
        ax2.yaxis.offsetText.set_color(colors[color_2])

        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_edgecolor(colors[color_2])

        # plt.ylim([0, 1.05])
        ax.tick_params(axis="y", color=colors[color], labelcolor=colors[color])
        ax.spines["left"].set_edgecolor(colors[color])
        ax.set_ylabel(ylabel + ", mean", color=colors[color], fontsize=labelsize)

        ax2.set_xlim([min(time), max(time)])
        ax.set_xlim([min(time), max(time)])

        ax.legend(["Mean", "90% prediction interval"], loc=loc1)
        plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.folder,
                                     feature + "_prediction-interval_sensitivity_" + sensitivity + self.figureformat))

        if show:
            plt.show()
        else:
            plt.close()

        reset_style()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot data")
#     parser.add_argument("-d", "--data_dir",
#                         help="Directory the data is stored in", default="data")
#     parser.add_argument("-o", "--folder",
#                         help="Folders to find compare files", default="figures")

#     args = parser.parse_args()

#     figureformat = ".png"
