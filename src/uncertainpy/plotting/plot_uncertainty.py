import glob
# import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from .prettyplot import prettyPlot, prettyBar
from .prettyplot import spines_color, get_current_colormap
from .prettyplot import get_colormap_tableu20, set_style
from .prettyplot import axis_grey, labelsize, fontsize, titlesize

from ..data import Data
from ..utils import create_logger


# TODO find a good way to find the directory where the data files are

# TODO compare plots in a grid of all plots,
# such as plotting all features in a grid plot

# TODO Change the use of **plot_kwargs to use a dict for specific plotting commands?



class PlotUncertainty(object):
    def __init__(self,
                 filename=None,
                 output_dir="figures/",
                 figureformat=".png",
                 verbose_level="info",
                 verbose_filename=None):

        self._output_dir = None

        self.output_dir = output_dir
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
    def output_dir(self):
        return self._output_dir


    @output_dir.setter
    def output_dir(self, new_output_dir):
        self._output_dir = new_output_dir

        if not os.path.isdir(new_output_dir):
            os.makedirs(new_output_dir)


    def str_to_latex(self, text):
        if "_" in text:
            txt = text.split("_")
            return "$" + txt[0] + "_{" + "-".join(txt[1:]) + "}$"
        else:
            return text


    def list_to_latex(self, texts):
        tmp = []
        for txt in texts:
            tmp.append(self.str_to_latex(txt))

        return tmp


    def simulator_results(self, foldername="simulator_results"):
        if self.data.ndim(self.data.model_name) == 0:
            self.simulator_results_0d(foldername=foldername)

        elif self.data.ndim(self.data.model_name) == 1:
            self.simulator_results_1d(foldername=foldername)

        elif self.data.ndim(self.data.model_name) == 2:
            self.simulator_results_2d(foldername=foldername)
        else:
            raise NotImplementedError(">2D plots not implementes")


    # TODO does not have a test
    def simulator_results_0d(self, foldername="simulator_results", **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if self.data.model_name not in self.data or "U"  not in self.data[self.data.model_name]:
            self.logger.warning("No model results to plot.")
            return

        if self.data.ndim(self.data.model_name) != 0:
            raise ValueError("{} is not a 0D feature".format(self.data.model_name))

        save_folder = os.path.join(self.output_dir, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        prettyPlot(self.data[self.data.model_name]["U"],
                   xlabel=r"simulator run \#number",
                   ylabel=self.data.get_labels(self.data.model_name)[0],
                   title="{}, simulator result".format(self.data.model_name.replace("_", " ")),
                   new_figure=True,
                   **plot_kwargs)
        plt.savefig(os.path.join(save_folder, "U" + self.figureformat))
        plt.close()


    def simulator_results_1d(self, foldername="simulator_results", **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if self.data.model_name not in self.data or "U"  not in self.data[self.data.model_name]:
            self.logger.warning("No model results to plot.")
            return

        if self.data.ndim(self.data.model_name) != 1:
            raise ValueError("{} is not a 1D feature".format(self.data.model_name))

        i = 1
        save_folder = os.path.join(self.output_dir, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        labels = self.data.get_labels(self.data.model_name)
        xlabel, ylabel = labels

        padding = len(str(len(self.data[self.data.model_name]["U"]) + 1))
        for U in self.data[self.data.model_name]["U"]:
            prettyPlot(self.data[self.data.model_name]["t"], U,
                       xlabel=xlabel, ylabel=ylabel,
                       title="{}, simulator result {:d}".format(self.data.model_name.replace("_", " "), i), new_figure=True, **plot_kwargs)
            plt.savefig(os.path.join(save_folder,
                                     "U_{0:0{1}d}".format(i, padding) + self.figureformat))
            plt.close()
            i += 1



    # TODO double check ylabel ans zlabel
    def simulator_results_2d(self, foldername="simulator_results", **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if self.data.model_name not in self.data or "U"  not in self.data[self.data.model_name]:
            self.logger.warning("No model results to plot.")
            return

        if self.data.ndim(self.data.model_name) != 2:
            raise ValueError("{} is not a 2D feature.".format(self.data.model_name))


        i = 1
        save_folder = os.path.join(self.output_dir, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        labels = self.data.get_labels(self.data.model_name)
        xlabel, ylabel, zlabel = labels


        padding = len(str(len(self.data[self.data.model_name]["U"]) + 1))
        for U in self.data[self.data.model_name]["U"]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("{}, simulator result {:d}".format(self.data.model_name.replace("_", " "), i))

            iax = ax.imshow(U, cmap="viridis", aspect="auto",
                            extent=[self.data[self.data.model_name]["t"][0],
                                    self.data[self.data.model_name]["t"][-1],
                                    0, U.shape[0]],
                            **plot_kwargs)

            cbar = fig.colorbar(iax)
            cbar.ax.set_title(zlabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.savefig(os.path.join(save_folder,
                                     "U_{0:0{1}d}".format(i, padding) + self.figureformat))
            plt.close()
            i += 1



    def attribute_feature_1d(self,
                             feature=None,
                             attribute="E",
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

        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))

        if attribute not in self.data[feature]:
            msg = "{attribute_name} of {feature} does not exist. Unable to plot {attribute_name}"
            self.logger.warning(msg.format(attribute_name=attribute, feature=feature))
            return

        if "t" not in self.data[feature] or np.all(np.isnan(self.data[feature]["t"])):
            t = np.arange(0, len(self.data[feature][attribute]))
        else:
            t = self.data[feature]["t"]


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature + ", " + attribute_name
        prettyPlot(t, self.data[feature][attribute],
                   title.replace("_", " "), xlabel, ylabel, **plot_kwargs)


        save_name = feature + "_" + attribute_name

        if hardcopy:
            plt.savefig(os.path.join(self.output_dir,
                                     save_name + self.figureformat),
                        bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()



    def attribute_feature_2d(self,
                             feature=None,
                             attribute="E",
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

        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        if attribute not in self.data[feature]:
            msg = "{attribute_name} of {feature} does not exist. Unable to plot {attribute_name}"
            self.logger.warning(msg.format(attribute_name=attribute, feature=feature))
            return

        if "t" not in self.data[feature] or np.all(np.isnan(self.data[feature]["t"])):
            extent = None
        else:
            extent=[self.data[feature]["t"][0], self.data[feature]["t"][-1],
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
        cbar.ax.set_title(zlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        save_name = feature + "_" + attribute_name

        if hardcopy:
            plt.savefig(os.path.join(self.output_dir,
                                     save_name + self.figureformat),
                        bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()



    def mean_1d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_1d(feature,
                                  attribute="E",
                                  attribute_name="mean",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)


    def variance_1d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_1d(feature,
                                  attribute="Var",
                                  attribute_name="variance",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)

    def mean_2d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_2d(feature,
                                  attribute="E",
                                  attribute_name="mean",
                                  hardcopy=hardcopy,
                                  show=show,
                                  **plot_kwargs)


    def variance_2d(self, feature, hardcopy=True, show=False, **plot_kwargs):
        self.attribute_feature_2d(feature,
                                  attribute="Var",
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

        if "E" not in self.data[feature] or "Var" not in self.data[feature]:
            msg = "Mean and/or variance of {feature} does not exist. ".format(feature=feature) \
                    + "Unable to plot mean and variance"
            self.logger.warning(msg)
            return


        if "t" not in self.data[feature] or np.all(np.isnan(self.data[feature]["t"])):
            t = np.arange(0, len(self.data[feature]["E"]))
        else:
            t = self.data[feature]["t"]


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature + ", mean and variance"
        ax = prettyPlot(t, self.data[feature]["E"],
                        title.replace("_", " "), xlabel, ylabel + ", mean",
                        style=style, **plot_kwargs)

        colors = get_current_colormap()

        ax2 = ax.twinx()

        spines_color(ax2, edges={"top": "None", "bottom": "None",
                                 "right": colors[color+2], "left": "None"})
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=colors[color+2], labelcolor=colors[color+2], labelsize=labelsize)
        ax2.set_ylabel(ylabel + ', variance', color=colors[color+2], fontsize=labelsize)

        # ax2.set_xlim([min(self.data.t[feature]), max(self.data.t[feature])])
        # ax2.set_ylim([min(self.data.Var[feature]), max(self.data.Var[feature])])

        ax2.plot(t, self.data[feature]["Var"],
                 color=colors[color+2], linewidth=2, antialiased=True)

        ax2.yaxis.offsetText.set_fontsize(16)
        ax2.yaxis.offsetText.set_color(colors[color+2])


        ax.tick_params(axis="y", color=colors[color], labelcolor=colors[color])
        ax.spines["left"].set_edgecolor(colors[color])

        ax.set_ylabel(ylabel + ', mean', color=colors[color], fontsize=16)


        if hardcopy:
            plt.savefig(os.path.join(self.output_dir,
                                     feature + "_mean-variance" + self.figureformat),
                        bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()
        #
        # if not show or not hardcopy:
        #     return ax, ax2



    def confidence_interval_1d(self,
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

        if "E" not in self.data[feature] \
            or "p_05" not in self.data[feature] \
                or "p_95" not in self.data[feature]:
            msg = "E, p_05  and/or p_95 of {feature} does not exist. Unable to plot confidence interval"
            self.logger.warning(msg.format(feature=feature))
            return


        if "t" not in self.data[feature] or np.all(np.isnan(self.data[feature]["t"])):
            t = np.arange(0, len(self.data[feature]["E"]))
        else:
            t = self.data[feature]["t"]


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature.replace("_", " ") + ", 90\\% confidence interval"
        prettyPlot(t, self.data[feature]["E"], title=title,
                   xlabel=xlabel, ylabel=ylabel, color=0,
                   **plot_kwargs)

        colors = get_current_colormap()
        plt.fill_between(t,
                         self.data[feature]["p_05"],
                         self.data[feature]["p_95"],
                         alpha=0.5, color=colors[0])


        plt.legend(["mean", "90\% confidence interval"])


        if hardcopy:
            plt.savefig(os.path.join(self.output_dir,
                                     feature + "_confidence-interval" + self.figureformat),
                        bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


    def sensitivity_1d(self,
                       feature=None,
                       sensitivity="sensitivity_1",
                       hardcopy=True,
                       show=False,
                       **plot_kwargs):

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

        if "t" not in self.data[feature] or np.all(np.isnan(self.data[feature]["t"])):
            t = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            t = self.data[feature]["t"]

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        for i in range(len(self.data[feature][sensitivity])):
            prettyPlot(t, self.data[feature][sensitivity][i],
                       title=feature.replace("_", " ") + ", " + sensitivity.replace("_", " ") + ", " + self.str_to_latex(self.data.uncertain_parameters[i]),
                       xlabel=xlabel, ylabel="sensitivity",
                       color=i,
                       nr_colors=len(self.data.uncertain_parameters), **plot_kwargs)
            # plt.ylim([0, 1.05])


            if hardcopy:
                plt.savefig(os.path.join(self.output_dir,
                                         feature + "_" + sensitivity + "_"
                                         + self.data.uncertain_parameters[i] + self.figureformat),
                            bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()




    def sensitivity_1d_grid(self,
                            feature=None,
                            sensitivity="sensitivity_1",
                            hardcopy=True,
                            show=False,
                            **plot_kwargs):
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

        if "t" not in self.data[feature] or np.all(np.isnan(self.data[feature]["t"])):
            t = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            t = self.data[feature]["t"]

        parameter_names = self.data.uncertain_parameters

        # get size of the grid in x and y directions
        nr_plots = len(parameter_names)
        grid_size = np.ceil(np.sqrt(nr_plots))
        grid_x_size = int(grid_size)
        grid_y_size = int(np.ceil(nr_plots/float(grid_x_size)))

        set_style("seaborn-darkgrid")
        fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size, squeeze=False)

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        # Add a larger subplot to use to set a common xlabel and ylabel
        set_style("seaborn-white")
        ax = fig.add_subplot(111, zorder=-10)
        spines_color(ax, edges={"top": "None", "bottom": "None",
                                "right": "None", "left": "None"})
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('sensitivity')

        for i in range(0, grid_x_size*grid_y_size):
            nx = i % grid_x_size
            ny = int(np.floor(i/float(grid_x_size)))

            ax = axes[ny][nx]

            if i < nr_plots:
                prettyPlot(t, self.data[feature][sensitivity][i],
                           title=self.str_to_latex(parameter_names[i]), color=i,
                           nr_colors=nr_plots, ax=ax,
                           **plot_kwargs)

                for tick in ax.get_xticklabels():
                    tick.set_rotation(-30)

                ax.set_ylim([0, 1.05])
                # ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)
                ax.tick_params(labelsize=10)
            else:
                ax.axis("off")

        title = feature.replace("_", " ") + ", " + sensitivity.replace("_", " ")
        plt.suptitle(title, fontsize=titlesize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)


        if hardcopy:
            plt.savefig(os.path.join(self.output_dir,
                                     feature + "_" + sensitivity + "_grid" + self.figureformat),
                        bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()




    def sensitivity_1d_combined(self,
                                feature=None,
                                sensitivity="sensitivity_1",
                                hardcopy=True,
                                show=False,
                                **plot_kwargs):
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

        if "t" not in self.data[feature] or np.all(np.isnan(self.data[feature]["t"])):
            t = np.arange(0, len(self.data[feature][sensitivity][0]))
        else:
            t = self.data[feature]["t"]


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        for i in range(len(self.data[feature][sensitivity])):
            prettyPlot(t,
                       self.data[feature][sensitivity][i],
                       title=feature.replace("_", " ") + ", " + sensitivity.replace("_", " "),
                       xlabel=xlabel,
                       ylabel="sensitivity",
                       new_figure=False,
                       color=i,
                       nr_colors=len(self.data.uncertain_parameters),
                       label=self.str_to_latex(self.data.uncertain_parameters[i]),
                       **plot_kwargs)

        plt.ylim([0, 1.05])
        if len(self.data[feature][sensitivity]) > 4:
            plt.xlim([t[0], 1.3*t[-1]])

        plt.legend()

        if hardcopy:
            plt.savefig(os.path.join(self.output_dir,
                                     feature + "_" + sensitivity + self.figureformat),
                        bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


    def features_1d(self, sensitivity="sensitivity_1"):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.mean_1d(feature=feature)
                self.variance_1d(feature=feature)
                self.mean_variance_1d(feature=feature)
                self.confidence_interval_1d(feature=feature)

                if sensitivity in self.data[feature]:
                    self.sensitivity_1d(feature=feature, sensitivity=sensitivity)
                    self.sensitivity_1d_combined(feature=feature, sensitivity=sensitivity)
                    self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity)


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
                   sensitivity="sensitivity_1",
                   hardcopy=True,
                   show=False):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if self.data.ndim(feature) != 0:
            raise ValueError("{} is not a 1D feature".format(feature))

        for data_type in ["E", "Var", "p_05", "p_95"]:
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

        values = [self.data[feature]["E"], self.data[feature]["Var"],
                  self.data[feature]["p_05"], self.data[feature]["p_95"]]

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
            ax2.set_ylabel('sensitivity', fontsize=fontsize)
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
                       self.list_to_latex(self.data.uncertain_parameters),
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
            plt.savefig(os.path.join(self.output_dir, save_name))

        if show:
            plt.show()
        else:
            plt.close()


        return ax


    def total_sensitivity(self,
                          feature,
                          sensitivity="sensitivity_1",
                          hardcopy=True,
                          show=False):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        if feature not in self.data:
            raise ValueError("{} is not a feature".format(feature))

        if "total_" + sensitivity not in self.data[feature]:
            msg = "{sensitivity} of {feature} does not exist. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        width = 0.2

        index = np.arange(1, len(self.data.uncertain_parameters)+1)*width


        prettyBar(self.data[feature]["total_" + sensitivity],
                  title="total " + sensitivity.replace("_", " ")
                  + ", " + feature.replace("_", " "),
                  xlabels=self.list_to_latex(self.data.uncertain_parameters),
                  ylabel="\% total sensitivity",
                  nr_colors=len(self.data.uncertain_parameters),
                  index=index)


        plt.ylim([0, 1])

        save_name = feature + "_total-" + sensitivity + self.figureformat

        if hardcopy:
            plt.savefig(os.path.join(self.output_dir, save_name))

        if show:
            plt.show()
        else:
            plt.close()


    def total_sensitivity_all(self,
                              sensitivity="sensitivity_1",
                              hardcopy=True,
                              show=False):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        for feature in self.data:
            if "total_" + sensitivity in self.data[feature]:
                self.total_sensitivity(feature=feature,
                                    sensitivity=sensitivity,
                                    hardcopy=hardcopy,
                                    show=show)


    def features_0d(self, sensitivity="sensitivity_1", hardcopy=True, show=False):
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



    # def plot_allNoSensitivity(self, sensitivity="sensitivity_1"):
    #     if self.data is None:
    #         raise ValueError("Datafile must be loaded.")
    #
    #
    #     self.features_1d(sensitivity=sensitivity)
    #     self.features_0d(sensitivity=sensitivity)


    def plot_all(self, sensitivity="sensitivity_1"):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        self.features_2d()
        self.features_1d(sensitivity=sensitivity)
        self.features_0d(sensitivity=sensitivity)

        if sensitivity is not None:
            self.total_sensitivity_all(sensitivity=sensitivity)
            self.total_sensitivity_grid(sensitivity=sensitivity)



    # TODO find a more descriptive name
    def plot_all_sensitivities(self):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")


        self.plot_all(sensitivity="sensitivity_1")

        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.sensitivity_1d(feature=feature, sensitivity="sensitivity_t")
                self.sensitivity_1d_combined(feature=feature, sensitivity="sensitivity_t")
                self.sensitivity_1d_grid(feature=feature, sensitivity="sensitivity_t")

        self.features_0d(sensitivity="sensitivity_t")

        self.total_sensitivity_all(sensitivity="sensitivity_t")
        self.total_sensitivity_grid(sensitivity="sensitivity_t")


    def plot_condensed(self, sensitivity="sensitivity_1"):
        for feature in self.data:
            if self.data.ndim(feature) == 1:
                self.mean_variance_1d(feature=feature)
                self.confidence_interval_1d(feature=feature)

                if sensitivity in self.data[feature]:
                    self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity)

        self.features_0d(sensitivity=sensitivity)
        self.features_2d()

        if sensitivity is not None:
            self.total_sensitivity_grid(sensitivity=sensitivity)




    def plot(self, condensed=True, sensitivity="sensitivity_1"):
        if condensed:
            self.plot_condensed(sensitivity=sensitivity)
        else:
            if sensitivity is None:
                self.plot_all(sensitivity)
            else:
                self.plot_all_sensitivities()

    # def plot_allFromExploration(self, exploration_folder):
    #     self.logger.info("Plotting all data")
    #
    #     original_data_dir = self.data_dir
    #     original_output_dir = self.output_dir
    #
    #     for folder in glob.glob(os.path.join(self.data_dir, "*")):
    #         self.data_dir = os.path.join(original_data_dir, folder.split("/")[-1])
    #         self.output_dir = os.path.join(original_output_dir,
    #                                                folder.split("/")[-1])
    #
    #         for filename in glob.glob(os.path.join(folder, "*")):
    #
    #             self.loadData(filename.split("/")[-1])
    #
    #         self.plot_all()
    #
    #     self.data_dir = original_data_dir
    #     self.output_dir = original_output_dir
    #




    def total_sensitivity_grid(self,
                               sensitivity="sensitivity_1",
                               hardcopy=True,
                               show=False,
                               **plot_kwargs):
        if self.data is None:
            raise ValueError("Datafile must be loaded.")

        no_sensitivity = True
        for feature in self.data:
            if "total_" + sensitivity in self.data[feature]:
                no_sensitivity = False

        if no_sensitivity:
            msg = "All total_{sensitivity}s are missing. Unable to plot total_{sensitivity}_grid"
            self.logger.warning(msg.format(sensitivity=sensitivity))
            return

        # get size of the grid in x and y directions
        nr_plots = len(self.data)
        grid_size = np.ceil(np.sqrt(nr_plots))
        grid_x_size = int(grid_size)
        grid_y_size = int(np.ceil(nr_plots/float(grid_x_size)))

        # plt.close("all")


        set_style("seaborn-dark")
        fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size, squeeze=False)
        set_style("seaborn-white")


        # Add a larger subplot to use to set a common xlabel and ylabel

        ax = fig.add_subplot(111, zorder=-10)
        spines_color(ax, edges={"top": "None", "bottom": "None",
                                "right": "None", "left": "None"})
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        # ax.set_xlabel(self.data.xlabel)
        ax.set_ylabel('\% total ' + sensitivity.split("_")[0] + " " + sensitivity.split("_")[1])

        width = 0.2
        index = np.arange(1, len(self.data.uncertain_parameters)+1)*width


        for i in range(0, grid_x_size*grid_y_size):
            nx = i % grid_x_size
            ny = int(np.floor(i/float(grid_x_size)))

            ax = axes[ny][nx]

            if i < nr_plots:
                if "total_" + sensitivity not in self.data[self.data.keys()[i]]:
                    msg = "total_{sensitivity} of {feature} is None. Unable to plot total_{sensitivity}_grid"
                    self.logger.warning(msg.format(sensitivity=sensitivity,
                                                   feature=self.data.keys()[i]))
                    ax.axis("off")
                    continue

                prettyBar(self.data[self.data.keys()[i]]["total_" + sensitivity],
                          title=self.data.keys()[i].replace("_", " "),
                          xlabels=self.list_to_latex(self.data.uncertain_parameters),
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

        title = "total " + sensitivity.replace("_", " ")
        plt.suptitle(title, fontsize=titlesize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)


        if hardcopy:
            plt.savefig(os.path.join(self.output_dir,
                                     "total-" + sensitivity + "_grid" + self.figureformat),
                        bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot data")
#     parser.add_argument("-d", "--data_dir",
#                         help="Directory the data is stored in", default="data")
#     parser.add_argument("-o", "--output_dir",
#                         help="Folders to find compare files", default="figures")

#     args = parser.parse_args()

#     figureformat = ".png"


    # plot = PlotUncertainty(data_dir=args.data_dir,
    #                        output_dir=args.output_dir,
    #                        figureformat=figureformat)
    #
    # # plot.plot_all()
    # plot.plot_allFromExploration()

    # sortByParameters(path=output_dir, outputpath=output_dir)
