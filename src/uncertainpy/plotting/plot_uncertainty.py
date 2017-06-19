import glob
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ..data import Data
from ..utils import create_logger

from prettyplot import prettyPlot, prettyBar
from prettyplot import spines_color, get_current_colormap
from prettyplot import get_colormap_tableu20, set_style
from prettyplot import axis_grey, labelsize, fontsize, titlesize


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
        if self.data.model_name in self.data.features_0d:
            self.simulator_results_0d(foldername=foldername)
        elif self.data.model_name in self.data.features_1d:
            self.simulator_results_1d(foldername=foldername)
        else:
            self.simulator_results_2d(foldername=foldername)


    # TODO does not have a test
    def simulator_results_0d(self, foldername="simulator_results", **plot_kwargs):
        if self.data.model_name not in self.data.features_0d:
            raise ValueError("{} is not a 0D feature".format(self.data.model_name))

        save_folder = os.path.join(self.output_dir, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        prettyPlot(self.data.U[self.data.model_name],
                   xlabel="simulator run \#number",
                   ylabel=self.data.get_labels(self.data.model_name)[0],
                   title="{}, simulator result".format(self.data.model_name.replace("_", " ")),
                   new_figure=True,
                   **plot_kwargs)
        plt.savefig(os.path.join(save_folder, "U" + self.figureformat))


    def simulator_results_1d(self, foldername="simulator_results", **plot_kwargs):
        if self.data.model_name not in self.data.features_1d:
            raise ValueError("{} is not a 1D feature".format(self.data.model_name))

        i = 1
        save_folder = os.path.join(self.output_dir, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        labels = self.data.get_labels(self.data.model_name)
        xlabel, ylabel = labels

        padding = len(str(len(self.data.U[self.data.model_name]) + 1))
        for U in self.data.U[self.data.model_name]:
            prettyPlot(self.data.t[self.data.model_name], U,
                       xlabel=xlabel, ylabel=ylabel,
                       title="{}, simulator result {:d}".format(self.data.model_name.replace("_", " "), i), new_figure=True, **plot_kwargs)
            plt.savefig(os.path.join(save_folder,
                                     "U_{0:0{1}d}".format(i, padding) + self.figureformat))
            plt.clf()
            i += 1


    # TODO double check ylabel ans zlabel
    def simulator_results_2d(self, foldername="simulator_results", **plot_kwargs):
        if self.data.model_name not in self.data.features_2d:
            raise ValueError("{} is not a 2D feature".format(self.data.model_name))

        i = 1
        save_folder = os.path.join(self.output_dir, foldername)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        labels = self.data.get_labels(self.data.model_name)
        xlabel, ylabel, zlabel = labels


        padding = len(str(len(self.data.U[self.data.model_name]) + 1))
        for U in self.data.U[self.data.model_name]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("{}, simulator result {:d}".format(self.data.model_name.replace("_", " "), i))

            iax = ax.imshow(U, cmap="viridis", aspect="auto",
                            extent=[self.data.t[self.data.model_name][0],
                                    self.data.t[self.data.model_name][-1],
                                    0, U.shape[0]],
                            **plot_kwargs)

            cbar = fig.colorbar(iax)
            cbar.ax.set_title(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(zlabel)
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
            raise ValueError("Datafile must be loaded")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        value = getattr(self.data, attribute)[feature]
        t = self.data.t[feature]

        if value is None:
            msg = "{attribute_name} of {feature} is None. Unable to plot {attribute_name}"
            self.logger.warning(msg.format(attribute_name=attribute_name, feature=feature))
            return

        if np.all(np.isnan(t)):
            t = np.arange(0, len(value))


        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature + ", " + attribute_name
        prettyPlot(t, value,
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
            raise ValueError("Datafile must be loaded")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data.features_2d:
            raise ValueError("%s is not a 2D feature" % (feature))

        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        value = getattr(self.data, attribute)

        if value[feature] is None:
            msg = "{attribute_name} of {feature} is None. Unable to plot {attribute_name}"
            self.logger.warning(msg.format(attribute_name=attribute_name, feature=feature))
            return

        title = feature + ", " + attribute_name
        labels = self.data.get_labels(feature)
        xlabel, ylabel, zlabel = labels

        if np.all(np.isnan(self.data.t[feature])):
            extent = None
        else:
            extent=[self.data.t[feature][0], self.data.t[feature][-1],
                    0, value[feature].shape[0]]


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title.replace("_", " "))

        iax = ax.imshow(value[feature], cmap="viridis", aspect="auto",
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
            raise ValueError("Datafile must be loaded")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        if self.data.E[feature] is None or self.data.Var[feature] is None:
            self.logger.warning("Mean and/or variance of {feature} is None.".format(feature=feature)
                                + "Unable to plot mean and variance")
            return

        t = self.data.t[feature]

        if np.all(np.isnan(t)):
            t = np.arange(0, len(self.data.E[feature]))

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature + ", mean and variance"
        ax = prettyPlot(t, self.data.E[feature],
                        title.replace("_", " "), xlabel, ylabel + ", mean",
                        style=style, **plot_kwargs)

        colors = get_current_colormap()

        ax2 = ax.twinx()

        spines_color(ax2, edges={"top": "None", "bottom": "None",
                                 "right": colors[color+1], "left": "None"})
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=colors[color+1], labelcolor=colors[color+1], labelsize=labelsize)
        ax2.set_ylabel(ylabel + ', variance', color=colors[color+1], fontsize=labelsize)

        # ax2.set_xlim([min(self.data.t[feature]), max(self.data.t[feature])])
        # ax2.set_ylim([min(self.data.Var[feature]), max(self.data.Var[feature])])

        ax2.plot(t, self.data.Var[feature],
                 color=colors[color+1], linewidth=2, antialiased=True)

        ax2.yaxis.offsetText.set_fontsize(16)
        ax2.yaxis.offsetText.set_color(colors[color+1])


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
            raise ValueError("Datafile must be loaded")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        if self.data.E[feature] is None \
            or self.data.p_05[feature] is None \
                or self.data.p_95[feature] is None:
            msg = "Mean, p_05  and/or p_95 of {feature} is NaN. Unable to plot confidence interval"
            self.logger.warning(msg.format(feature=feature))
            return

        t = self.data.t[feature]

        if np.all(np.isnan(t)):
            t = np.arange(0, len(self.data.E[feature]))

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        title = feature.replace("_", " ") + ", 90\\% confidence interval"
        prettyPlot(t, self.data.E[feature], title=title,
                   xlabel=xlabel, ylabel=ylabel, color=0,
                   **plot_kwargs)

        colors = get_current_colormap()
        plt.fill_between(t,
                         self.data.p_05[feature],
                         self.data.p_95[feature],
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
            raise ValueError("Datafile must be loaded")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        sense = getattr(self.data, sensitivity)

        if feature not in sense:
            msg = "{feature} not in {sensitivity}. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if sense[feature] is None:
            msg = "{sensitivity} of {feature} is None. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return


        t = self.data.t[feature]

        if np.all(np.isnan(t)):
            t = np.arange(0, len(sense[feature][0]))

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        for i in range(len(sense[feature])):
            prettyPlot(t, sense[feature][i],
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
            raise ValueError("Datafile must be loaded")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        sense = getattr(self.data, sensitivity)

        if feature not in sense:
            msg = "{feature} not in {sensitivity}. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if sense[feature] is None:
            msg = "{sensitivity} of {feature} is None. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return


        t = self.data.t[feature]

        if np.all(np.isnan(t)):
            t = np.arange(0, len(sense[feature][0]))


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
                prettyPlot(t, sense[feature][i],
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
            raise ValueError("Datafile must be loaded")

        if feature is None:
            feature = self.data.model_name

        if feature not in self.data.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        sense = getattr(self.data, sensitivity)

        if feature not in sense:
            msg = "{feature} not in {sensitivity}. Unable to plot {sensitivity} combined"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if sense[feature] is None:
            msg = "{sensitivity} of {feature} is None. Unable to plot {sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        labels = self.data.get_labels(feature)
        xlabel, ylabel = labels

        t = self.data.t[feature]

        if np.all(np.isnan(t)):
            t = np.arange(0, len(sense[feature][0]))

        for i in range(len(sense[feature])):
            prettyPlot(t,
                       sense[feature][i],
                       title=feature.replace("_", " ") + ", " + sensitivity.replace("_", " "),
                       xlabel=xlabel,
                       ylabel="sensitivity",
                       new_figure=False,
                       color=i,
                       nr_colors=len(self.data.uncertain_parameters),
                       label=self.str_to_latex(self.data.uncertain_parameters[i]),
                       **plot_kwargs)

        plt.ylim([0, 1.05])
        if len(sense[feature]) > 4:
            plt.xlim([self.data.t[feature][0], 1.3*self.data.t[feature][-1]])

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
        for feature in self.data.features_1d:
            self.mean_1d(feature=feature)
            self.variance_1d(feature=feature)
            self.mean_variance_1d(feature=feature)
            self.confidence_interval_1d(feature=feature)

            if sensitivity is not None:
                self.sensitivity_1d(feature=feature, sensitivity=sensitivity)
                self.sensitivity_1d_combined(feature=feature, sensitivity=sensitivity)
                self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity)


    def features_2d(self):
        for feature in self.data.features_2d:
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
            raise ValueError("Datafile must be loaded")

        if feature not in self.data.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))

        if sensitivity is None:
            sense = None
        else:
            sense = getattr(self.data, sensitivity)

        if self.data.E[feature] is None:
            msg = "Missing E for {feature}. Unable to plot"
            self.logger.warning(msg.format(feature=feature))
            return

        if self.data.Var[feature] is None:
            msg = "Missing Var for {feature}. Unable to plot"
            self.logger.warning(msg.format(feature=feature))
            return

        if self.data.p_05[feature] is None:
            msg = "Missing p_05 for {feature}. Unable to plot"
            self.logger.warning(msg.format(feature=feature))
            return

        if self.data.p_95[feature] is None:
            msg = "Missing p_95 for {feature}. Unable to plot"
            self.logger.warning(msg.format(feature=feature))
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

        values = [self.data.E[feature], self.data.Var[feature],
                  self.data.p_05[feature], self.data.p_95[feature]]

        ylabel = self.data.get_labels(feature)[0]

        ax = prettyBar(values,
                       index=xticks,
                       xlabels=xlabels,
                       ylabel=ylabel,
                       palette=get_colormap_tableu20())

        if sense is not None and sense[feature] is not None:
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

                l = ax2.bar(pos, sense[feature][i], width=width,
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
            raise ValueError("Datafile must be loaded")

        if feature not in self.data.feature_list:
            raise ValueError("%s is not a feature" % (feature))

        total_sense = getattr(self.data, "total_" + sensitivity)


        if feature not in total_sense:
            msg = "{feature} not in total_{sensitivity}. Unable to plot total_{sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        if total_sense[feature] is None:
            msg = "total_{sensitivity} of {feature} is None. Unable to plot total_{sensitivity}"
            self.logger.warning(msg.format(sensitivity=sensitivity, feature=feature))
            return

        width = 0.2

        index = np.arange(1, len(self.data.uncertain_parameters)+1)*width


        prettyBar(total_sense[feature],
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
            raise ValueError("Datafile must be loaded")

        for feature in self.data.feature_list:
            self.total_sensitivity(feature=feature,
                                   sensitivity=sensitivity,
                                   hardcopy=hardcopy,
                                   show=show)


    def features_0d(self, sensitivity="sensitivity_1", hardcopy=True, show=False):
        if self.data is None:
            raise ValueError("Datafile must be loaded")


        for feature in self.data.features_0d:
            self.feature_0d(feature, sensitivity=sensitivity, hardcopy=hardcopy, show=show)



    # TODO Not Tested
    def plot_folder(self, data_dir):
        self.logger.info("Plotting all data in folder")

        for f in glob.glob(os.path.join(data_dir, "*")):
            self.load(f.split(os.path.sep)[-1])

            self.plot_all()



    # def plot_allNoSensitivity(self, sensitivity="sensitivity_1"):
    #     if self.data is None:
    #         raise ValueError("Datafile must be loaded")
    #
    #
    #     self.features_1d(sensitivity=sensitivity)
    #     self.features_0d(sensitivity=sensitivity)


    def plot_all(self, sensitivity="sensitivity_1"):
        if self.data is None:
            raise ValueError("Datafile must be loaded")

        self.features_2d()
        self.features_1d(sensitivity=sensitivity)
        self.features_0d(sensitivity=sensitivity)

        if sensitivity is not None:
            self.total_sensitivity_all(sensitivity=sensitivity)
            self.total_sensitivity_grid(sensitivity=sensitivity)



    # TODO find a more descriptive name
    def plot_all_sensitivities(self):
        if self.data is None:
            raise ValueError("Datafile must be loaded")


        self.plot_all(sensitivity="sensitivity_1")

        for feature in self.data.features_1d:
            self.sensitivity_1d(feature=feature, sensitivity="sensitivity_t")
            self.sensitivity_1d_combined(feature=feature, sensitivity="sensitivity_t")
            self.sensitivity_1d_grid(feature=feature, sensitivity="sensitivity_t")

        self.features_0d(sensitivity="sensitivity_t")

        self.total_sensitivity_all(sensitivity="sensitivity_t")
        self.total_sensitivity_grid(sensitivity="sensitivity_t")


    def plot_condensed(self, sensitivity="sensitivity_1"):
        for feature in self.data.features_1d:
            self.features_2d()
            self.mean_variance_1d(feature=feature)
            self.confidence_interval_1d(feature=feature)

            if sensitivity is not None:
                self.sensitivity_1d_grid(feature=feature, sensitivity=sensitivity)

        self.features_0d(sensitivity=sensitivity)

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
            raise ValueError("Datafile must be loaded")

        total_sense = getattr(self.data, "total_" + sensitivity)


        no_sensitivity = True
        for feature in self.data.feature_list:
            if total_sense[feature] is not None:
                no_sensitivity = False

        if no_sensitivity:
            msg = "All total_{sensitivity}s are None. Unable to plot total_{sensitivity}_grid"
            self.logger.warning(msg.format(sensitivity=sensitivity))
            return

        # get size of the grid in x and y directions
        nr_plots = len(self.data.feature_list)
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
                if total_sense[self.data.feature_list[i]] is None:
                    msg = "total_{sensitivity} of {feature} is None. Unable to plot total_{sensitivity}_grid"
                    self.logger.warning(msg.format(sensitivity=sensitivity,
                                                   feature=self.data.feature_list[i]))
                    ax.axis("off")
                    continue

                prettyBar(total_sense[self.data.feature_list[i]],
                          title=self.data.feature_list[i].replace("_", " "),
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data")
    parser.add_argument("-d", "--data_dir",
                        help="Directory the data is stored in", default="data")
    parser.add_argument("-o", "--output_dir",
                        help="Folders to find compare files", default="figures")

    args = parser.parse_args()

    figureformat = ".png"


    # plot = PlotUncertainty(data_dir=args.data_dir,
    #                        output_dir=args.output_dir,
    #                        figureformat=figureformat)
    #
    # # plot.plot_all()
    # plot.plot_allFromExploration()

    # sortByParameters(path=output_dir, outputpath=output_dir)
