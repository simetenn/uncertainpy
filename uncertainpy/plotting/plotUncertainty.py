import os
import h5py
import sys
import shutil
import glob
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np

# Fix Remove import * once finished
from uncertainpy.plotting.prettyPlot import prettyPlot, prettyBar
from uncertainpy.plotting.prettyPlot import spines_edge_color, get_current_colormap
from uncertainpy.plotting.prettyPlot import set_legend, get_colormap_tableu20
from uncertainpy.plotting.prettyPlot import axis_grey, labelsize, fontsize
from uncertainpy.utils import create_logger

# TODO rewrite gif() to use less memory when creating GIF
# (Only load one dataset at the time)

# TODO Add feature plots to gif()

# TODO find a good way to find the directory where the data files are

# TODO test out if seaborn is a better package to use for plottintg

# TODO move load() to it's own class

# TODO compare plots in a grid of all plots,
# such as plotting all features in a grid plot

class PlotUncertainty():
    def __init__(self,
                 data_dir="data/",
                 output_dir_figures="figures/",
                 output_dir_gif="gifs/",
                 figureformat=".png",
                 verbose_level="info",
                 verbose_filename=None,):

        self.data_dir = data_dir
        self.output_dir_figures = output_dir_figures
        self.output_dir_gif = output_dir_gif
        self.figureformat = figureformat
        self.f = None

        self.tmp_gif_output = ".tmp_gif_output/"
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.features_in_combined_plot = 3

        self.loaded_flag = False
        self.loaded_compare_flag = False

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)




    def loadData(self, filename):
        self.filename = filename
        f = h5py.File(os.path.join(self.data_dir, self.filename), 'r')

        # TODO what to do if output folder and data folder is the same.
        # Two options create the figures in the same folder, or create a new
        # folder with _figures added to the name?

        # full_output_dir_figures = os.path.join(self.output_dir_figures, filename)
        # if os.path.isfile(full_output_dir_figures):
        #     self.full_output_dir_figures = self.output_dir_figures
        # else:
        #     self.full_output_dir_figures = self.full_output_dir_figures
        #     if not os.path.isdir(self.full_output_dir_figures):
        #         os.makedirs(self.full_output_dir_figures)


        self.full_output_dir_figures = os.path.join(self.output_dir_figures, filename)
        if os.path.isfile(self.full_output_dir_figures):
            self.full_output_dir_figures = self.full_output_dir_figures + "_figures"

        if not os.path.isdir(self.full_output_dir_figures):
            os.makedirs(self.full_output_dir_figures)

        self.t = {}
        self.U = {}
        self.E = {}
        self.Var = {}
        self.p_05 = {}
        self.p_95 = {}
        self.sensitivity = {}

        for feature in f.keys():
            self.U[feature] = f[feature]["U"][()]
            self.E[feature] = f[feature]["E"][()]
            self.Var[feature] = f[feature]["Var"][()]
            self.p_05[feature] = f[feature]["p_05"][()]
            self.p_95[feature] = f[feature]["p_95"][()]

            if "sensitivity" in f[feature].keys():
                self.sensitivity[feature] = f[feature]["sensitivity"][()]
            else:
                self.sensitivity[feature] = None


            if "t" in f[feature].keys():
                self.t[feature] = f[feature]["t"][()]
            # else:
            #     self.t[feature] = None

        self.features_0d, self.features_1d = self.sortFeatures(self.E)

        self.uncertain_parameters = f.attrs["uncertain parameters"]
        self.loaded_flag = True


    def setData(self, t, U, E, Var, p_05, p_95, uncertain_parameters,
                sensitivity, foldername=None):

        self.t = t
        self.U = U
        self.E = E
        self.Var = Var
        self.p_05 = p_05
        self.p_95 = p_95
        self.sensitivity = sensitivity
        self.uncertain_parameters = uncertain_parameters


        self.features_0d, self.features_1d = self.sortFeatures(self.E)

        if foldername is None:
            self.filename = ""
            self.full_output_dir_figures = self.output_dir_figures
        else:
            self.filename = foldername
            self.full_output_dir_figures = os.path.join(self.output_dir_figures, self.filename)


        if os.path.isfile(self.full_output_dir_figures):
            self.full_output_dir_figures = self.full_output_dir_figures + "_figures"

        if not os.path.isdir(self.full_output_dir_figures):
            os.makedirs(self.full_output_dir_figures)


        self.loaded_flag = True


    def sortFeatures(self, results):
        features_1d = []
        features_0d = []

        for feature in results:
            if hasattr(results[feature], "__iter__"):
                if len(results[feature].shape) == 0:
                    features_0d.append(feature)
                elif len(results[feature].shape) == 1:
                    features_1d.append(feature)
                else:
                    self.logger.warning("No support for more than 0d and 1d plotting.")

            else:
                features_0d.append(feature)

        return features_0d, features_1d


    def plotMean(self, feature="directComparison", hardcopy=True, show=False,
                 **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        prettyPlot(self.t[feature], self.E[feature],
                   "Mean, " + feature, "time", "voltage", **kwargs)


        # min_t = self.t[feature].min()
        # max_t = self.t[feature].max()
        #
        # min_E = self.E[feature].min()
        # max_E = self.E[feature].max()
        #
        # plt.xlim(min_t, max_t)
        # plt.ylim(min_E, max_E)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_mean" + self.figureformat),
                        bbox_inches="tight")
            if not show:
                plt.close()

        if show:
            plt.show()

        # return min_t, max_t, min_E, max_E



    def plotVariance(self, feature="directComparison",
                     hardcopy=True, show=False,
                     **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        prettyPlot(self.t[feature], self.Var[feature], "Variance, " + feature,
                   "time", "voltage", **kwargs)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_variance" + self.figureformat),
                        bbox_inches="tight")
            if not show:
                plt.close()

        if show:
            plt.show()



    def plotMeanAndVariance(self, feature="directComparison", new_figure=True,
                            hardcopy=True, show=False, color=0, **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        ax = prettyPlot(self.t[feature], self.E[feature],
                        "Mean and variance, " + feature, "time", "voltage, mean",
                        **kwargs)

        colors = get_current_colormap()

        ax2 = ax.twinx()

        spines_edge_color(ax2, edges={"top": "None", "bottom": "None",
                                      "right": colors[color+1], "left": "None"})
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=colors[color+1], labelcolor=colors[color+1], labelsize=labelsize)
        ax2.set_ylabel('voltage, variance', color=colors[color+1], fontsize=fontsize)

        # ax2.set_xlim([min(self.t[feature]), max(self.t[feature])])
        # ax2.set_ylim([min(self.Var[feature]), max(self.Var[feature])])

        ax2.plot(self.t[feature], self.Var[feature],
                 color=colors[color+1], linewidth=2, antialiased=True)

        ax2.yaxis.offsetText.set_fontsize(16)
        ax2.yaxis.offsetText.set_color(colors[color+1])


        ax.tick_params(axis="y", color=colors[color], labelcolor=colors[color])
        ax.spines["left"].set_edgecolor(colors[color])

        ax.set_ylabel('voltage, mean', color=colors[color], fontsize=16)


        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_mean-variance" + self.figureformat),
                        bbox_inches="tight")
            if not show:
                plt.close()

        if show:
            plt.show()
        #
        # if not show or not hardcopy:
        #     return ax, ax2



    def plotConfidenceInterval(self, feature="directComparison", hardcopy=True,
                               show=False, **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")

        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        prettyPlot(self.t[feature], self.E[feature], title="Confidence interval, " + feature,
                   xlabel="time", ylabel="voltage", color=0,
                   **kwargs)

        colors = get_current_colormap()
        plt.fill_between(self.t[feature], self.p_05[feature], self.p_95[feature],
                         alpha=0.5, color=colors[0])


        set_legend(["Mean", "90\% confidence interval"])



        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_confidence-interval" + self.figureformat),
                        bbox_inches="tight")
            if not show:
                plt.close()

        if show:
            plt.show()


    def plotSensitivity(self, feature="directComparison", hardcopy=True, show=False,
                        **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")

        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        if feature not in self.sensitivity or self.sensitivity[feature] is None:
            return

        parameter_names = self.uncertain_parameters

        for i in range(len(self.sensitivity[feature])):
            prettyPlot(self.t[feature], self.sensitivity[feature][i],
                       title="sensitivity, " + feature,
                       xlabel="time", ylabel="sensitivity",
                       new_figure=True, **kwargs)
            plt.ylim([0, 1.05])

            if hardcopy:
                plt.savefig(os.path.join(self.full_output_dir_figures,
                                         feature + "_sensitivity_" + parameter_names[i] + self.figureformat),
                            bbox_inches="tight")
                if not show:
                    plt.close()

            if show:
                plt.show()


    def plotSensitivityGrid(self, feature="directComparison",
                            hardcopy=True, show=False, **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")

        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        if feature not in self.sensitivity or self.sensitivity[feature] is None:
            return

        parameter_names = self.uncertain_parameters

        # get size of the grid in x and y directions
        nr_plots = len(parameter_names)
        grid_size = np.ceil(np.sqrt(nr_plots))
        grid_x_size = int(grid_size)
        grid_y_size = int(np.ceil(nr_plots/float(grid_x_size)))

        fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size)

        for i in range(0, nr_plots):
            nx = i % grid_x_size
            ny = int(np.floor(i/float(grid_x_size)))
            if grid_y_size == 1:
                ax = axes[nx]
            else:
                ax = axes[ny][nx]


            prettyPlot(self.t[feature], self.sensitivity[feature][i],
                       title=parameter_names[i], color=i, nr_hues=nr_plots,
                       xlabel="time", ylabel="sensitivity", ax=ax,
                       **kwargs)
            ax.set_ylim([0, 1.05])

        title = feature + ", sensitivity"
        plt.suptitle(title, fontsize=titlesize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)


        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_sensitivity_grid" + self.figureformat),
                        bbox_inches="tight")
            if not show:
                plt.close()

        if show:
            plt.show()



    def plotSensitivityCombined(self, feature="directComparison",
                                hardcopy=True, show=False, **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        if feature not in self.sensitivity or self.sensitivity[feature] is None:
            return

        parameter_names = self.uncertain_parameters

        new_figure = True
        for i in range(len(self.sensitivity[feature])):
            prettyPlot(self.t[feature], self.sensitivity[feature][i],
                       title="sensitivity, " + feature,
                       xlabel="time", ylabel="sensitivity",
                       new_figure=False, **kwargs)

        plt.ylim([0, 1.05])
        if len(self.sensitivity[feature]) > 4:
            plt.xlim([self.t[feature][0], 1.3*self.t[feature][-1]])

        set_legend(parameter_names)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_sensitivity" + self.figureformat),
                        bbox_inches="tight")
            if not show:
                plt.close()

        if show:
            plt.show()


    def plot1dFeatures(self):
        for feature in self.features_1d:
            self.plotMean(feature=feature)
            self.plotVariance(feature=feature)
            self.plotMeanAndVariance(feature=feature)
            self.plotConfidenceInterval(feature=feature)
            self.plotSensitivity(feature=feature)
            self.plotSensitivityCombined(feature=feature)
            self.plotSensitivityGrid(feature=feature)




    # TODO not finhised, missing correct label placement
    def plot0dFeature(self, feature, max_legend_size=5,
                      hardcopy=True, show=False):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))


        if len(self.uncertain_parameters) > max_legend_size:
            legend_size = max_legend_size
        else:
            legend_size = len(self.uncertain_parameters)

        legend_width = np.ceil(len(self.uncertain_parameters)/float(max_legend_size))

        width = 0.2
        distance = 0.5

        xlabels = ["Mean", "Variance", "$P_5$", "$P_{95}$"]
        xticks = [0, width, distance + width, distance + 2*width]

        values = [self.E[feature], self.Var[feature],
                  self.p_05[feature], self.p_95[feature]]


        ax = prettyBar(values, index=xticks, xlabels=xlabels, ylabel="Value",
                       palette=get_colormap_tableu20())
        # plt.show()

        if self.sensitivity[feature] is not None:
            pos = 2*distance + 2*width

            ax2 = ax.twinx()

            spines_edge_color(ax2, edges={"top": "None", "bottom": "None",
                                          "right": axis_grey, "left": "None"})
            ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                            color=axis_grey, labelsize=labelsize)
            ax2.set_ylabel('Sensitivity', fontsize=fontsize)
            ax2.set_ylim([0, 1.05])


            i = 0
            legend_bars = []
            colors = get_current_colormap()

            for parameter in self.uncertain_parameters:

                l = ax2.bar(pos, self.sensitivity[feature][i], width=width,
                            align='center', color=colors[4+i], linewidth=0)

                legend_bars.append(l)

                i += 1
                pos += width

            xticks.append(pos - width*i/2.)
            xlabels.append("Sensitivity")

            location = (0.5, 1.01 + legend_width*0.095)
            lgd = plt.legend(legend_bars, self.uncertain_parameters,
                             loc='upper center', bbox_to_anchor=location,
                             ncol=legend_size)
            lgd.get_frame().set_edgecolor(axis_grey)

            fig = plt.gcf()
            fig.subplots_adjust(top=(0.91 - legend_width*0.053))


        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=-45)


        title = self.filename + ", " + feature
        title = title.replace("_", "\_")
        plt.suptitle(title, fontsize=titlesize)

        save_name = feature + self.figureformat

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures, save_name))
            if not show:
                plt.close()

        if show:
            plt.show()

        return ax



    def plot0dFeatures(self, hardcopy=True, show=False):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        for feature in self.features_0d:
            self.plot0dFeature(feature, hardcopy=hardcopy, show=show)



    def plotAllDataInFolder(self):
        self.logger.info("Plotting all data in folder")

        for f in glob.glob(os.path.join(self.data_dir, "*")):
            self.loadData(f.split(os.path.sep)[-1])

            self.plotAllData()


    def plotAllData(self):
        self.plot1dFeatures()
        self.plot0dFeatures()




    def plotAllDataFromExploration(self):
        self.logger.info("Plotting all data")

        original_data_dir = self.data_dir
        original_output_dir_figures = self.output_dir_figures

        for folder in glob.glob(os.path.join(self.data_dir, "*")):
            self.data_dir = os.path.join(original_data_dir, folder.split("/")[-1])
            self.output_dir_figures = os.path.join(original_output_dir_figures,
                                                   folder.split("/")[-1])

            for filename in glob.glob(os.path.join(folder, "*")):

                self.loadData(filename.split("/")[-1])

                self.plot1dFeatures()

                self.plot0dFeatures()

        self.data_dir = original_data_dir
        self.output_dir_figures = original_output_dir_figures





    # TODO expand this to work with regex?
    def loadCompareData(self, filename, compare_folders=None):
        self.t_compare = {}
        # self.U_compare = {}
        self.E_compare = {}
        self.Var_compare = {}
        self.p_05_compare = {}
        self.p_95_compare = {}
        self.sensitivity_compare = {}

        if compare_folders is None:
            compare_folders = [folder for folder in os.listdir(self.data_dir)
                               if os.path.isdir(os.path.join(self.data_dir,
                                                             folder))]


        self.compare_folders = compare_folders
        for folder in self.compare_folders:
            name = folder.split(os.path.sep)[-1]

            self.loadData(os.path.join(name, filename))

            self.t_compare[name] = self.t
            # self.U_compare[name] = self.U
            self.E_compare[name] = self.E
            self.Var_compare[name] = self.Var
            self.p_05_compare[name] = self.p_05
            self.p_95_compare[name] = self.p_95
            self.sensitivity_compare[name] = self.sensitivity


        self.compare_output_dir_figures = os.path.join(self.output_dir_figures, "compare")

        if not os.path.isdir(self.compare_output_dir_figures):
            os.makedirs(self.compare_output_dir_figures)

        self.loaded_compare_flag = True

    # def plotCompare(self, filename, reference_name="pc", compare_name="mc_"):
    #     self.logger.info("Plotting {} data".compare)
    #
    #     self.loadCompareData(filename, reference_name=reference_name,
    #                          compare_name=compare_name)


    def plotCompareFeature1d(self, feature, attribute="E"):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))



        output_dir_compare = os.path.join(self.output_dir_figures, "MC-compare")
        if not os.path.isdir(output_dir_compare):
            os.makedirs(output_dir_compare)






    def plotCompareMean(self, feature="directComparison",
                        hardcopy=True, show=False, **kwargs):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        max_values = []
        min_values = []
        legend = []
        new_figure = True

        for compare in self.compare_folders:
            min_values.append(self.E_compare[compare][feature].min())
            max_values.append(self.E_compare[compare][feature].max())

            legend.append(compare.replace("_", " "))

            self.t = self.t_compare[compare]
            self.E = self.E_compare[compare]

            self.plotMean(feature=feature, hardcopy=False, show=False,
                          new_figure=new_figure, nr_hues=len(self.compare_folders),
                          **kwargs)

            new_figure = False

        save_name = feature + "_mean_compare"

        plt.ylim([min(min_values)*0.99, max(max_values)*1.01])
        plt.legend(legend)

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()


    # TODO combine compare variance and compare mean into one general function
    # As with compare 0d features
    def plotCompareVariance(self, feature="directComparison",
                            hardcopy=True, show=False, **kwargs):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_values = []
        min_values = []
        legend = []
        new_figure = True

        for compare in self.compare_folders:
            min_values.append(self.Var_compare[compare][feature].min())
            max_values.append(self.Var_compare[compare][feature].max())

            legend.append(compare.replace("_", " "))

            self.t = self.t_compare[compare]
            self.Var = self.Var_compare[compare]

            self.plotVariance(feature=feature, hardcopy=False, show=False,
                              new_figure=new_figure,
                              nr_hues=len(self.compare_folders), **kwargs)

            new_figure = False
            color += 2

        save_name = feature + "_variance_compare"

        plt.ylim([min(min_values)*0.99, max(max_values)*1.01])
        plt.legend(legend)

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat), nr_hues=len(self.compare_folders))
            if not show:
                plt.close()

        if show:
            plt.show()



    def plotCompareMeanAndVariance(self, feature="directComparison",
                                   hardcopy=True, show=False, sns_style="dark",
                                   **kwargs):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_values_E = []
        min_values_E = []
        max_values_Var = []
        min_values_Var = []

        legend = []
        new_figure = True
        ax2 = None


        for compare in self.compare_folders:
            min_values_E.append(self.E_compare[compare][feature].min())
            max_values_E.append(self.E_compare[compare][feature].max())
            min_values_Var.append(self.Var_compare[compare][feature].min())
            max_values_Var.append(self.Var_compare[compare][feature].max())


            legend.append(compare.replace("_", " "))

            self.t = self.t_compare[compare]
            self.E = self.E_compare[compare]
            self.Var = self.Var_compare[compare]

            if new_figure:
                ax = prettyPlot(self.t[feature], self.E[feature],
                                "Mean and variance, " + feature, "time", "voltage, mean",
                                sns_style=sns_style, nr_hues=2*len(self.compare_folders),
                                new_figure=new_figure, **kwargs)

                colors = get_current_colormap()

                ax2 = ax.twinx()

            else:
                ax.plot(self.t[feature], self.E[feature],
                        color=colors[color], linewidth=2, antialiased=True,
                        zorder=3)


            ax2.plot(self.t[feature], self.Var[feature],
                     color=colors[color], linewidth=2, antialiased=True,
                     linestyle="--", zorder=3)


            new_figure = False
            color += 1





        spines_edge_color(ax2, edges={"top": "None", "bottom": "None",
                          "right": axis_grey, "left": "None"})
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=axis_grey, labelcolor="black", labelsize=labelsize)
        ax2.set_ylabel('voltage, variance', color="black", fontsize=labelsize)
        ax2.yaxis.offsetText.set_fontsize(labelsize)



        save_name = feature + "_mean-variance_compare"

        legend1 = ax.legend(legend, loc=2, title="Mean", fontsize=fontsize)
        legend2 = ax2.legend(legend, title="Variance", fontsize=fontsize)

        legend1.get_title().set_fontsize('18')
        legend2.get_title().set_fontsize('18')

        ax2.set_ylim([min(min_values_Var)*0.99, max(max_values_Var)*1.3])
        ax.set_ylim([min(min_values_E)*0.99, max(max_values_E)*1.3])


        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()



    def plotCompareConfidenceInterval(self, feature="directComparison",
                                      hardcopy=True, show=False, **kwargs):

        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_values = []
        min_values = []
        legend = []
        new_figure = True

        for compare in self.compare_folders:

            # legend.append(compare.replace("_", " "))
            legend.extend([compare.replace("_", " ") + ", Mean",
                           compare.replace("_", " ") + ", 90\% CI"])

            self.t = self.t_compare[compare]
            self.E = self.E_compare[compare]
            self.p_05 = self.p_05_compare[compare]
            self.p_95 = self.p_95_compare[compare]


            if new_figure:
                ax = prettyPlot(self.t[feature], self.E[feature], title="90\% Confidence interval, " + feature,
                                xlabel="time", ylabel="voltage", color=color,
                                **kwargs)

                colors = get_current_colormap()
            else:
                ax.plot(self.t[feature], self.E[feature],
                        color=colors[color], linewidth=2, antialiased=True,
                        zorder=3)

            plt.fill_between(self.t[feature], self.p_05[feature], self.p_95[feature],
                             alpha=0.5, color=colors[color])



            min_values.append(min([min(self.p_95[feature]),
                                   min(self.p_05[feature]),
                                   min(self.E[feature])]))
            max_values.append(max([max(self.p_95[feature]),
                                   max(self.p_05[feature]),
                                   max(self.E[feature])]))




            new_figure = False
            color += 1

        save_name = feature + "_confidence-interval_compare"

        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])
        plt.legend(legend, ncol=2)

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()



    # TODO not tested since MC currently does not calculate sensitivity
    def plotCompareSensitivity(self, feature="directComparison",
                               hardcopy=True, show=False, **kwargs):
            if feature not in self.sensitivity or self.sensitivity[feature] is None:
                return

            parameter_names = self.uncertain_parameters

            for i in range(len(self.sensitivity[feature])):
                legend = []
                new_figure = True

                for compare in self.compare_folders:

                    legend.append(compare.replace("_", " "))

                    self.t = self.t_compare[compare]
                    self.sensitivity = self.sensitivity_compare[compare]

                    prettyPlot(self.t[feature], self.sensitivity[feature][i],
                               title="sensitivity, " + feature,
                               xlabel="time", ylabel="sensitivity",
                               new_figure=new_figure, **kwargs)

                    new_figure = False

                plt.legend(legend)
                plt.ylim([0, 1.05])


                if hardcopy:
                    plt.savefig(os.path.join(self.full_output_dir_figures,
                                             feature + "_sensitivity_" + parameter_names[i]  + "_compare" + self.figureformat),
                                bbox_inches="tight")
                    if not show:
                        plt.close()

                if show:
                    plt.show()



    def plotCompare1dFeatures(self):
        for feature in self.features_1d:
            self.plotCompareMean(feature=feature)
            self.plotCompareVariance(feature=feature)
            self.plotCompareMeanAndVariance(feature=feature)
            self.plotCompareConfidenceInterval(feature=feature)




    def plotCompareAttributeFeature0d(self, feature, attribute="E", attribute_name="mean",
                                      hardcopy=True, show=False, **kwargs):
        if feature not in self.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))

        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        width = 0.2
        distance = 0.5

        values = []
        xlabels = []
        xticks = []
        pos = 0

        for compare in self.compare_folders:
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + 0.5*width)
            values.append(getattr(self, attribute + "_compare")[compare][feature])

            pos += distance + width

        prettyBar(values, index=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=len(self.compare_folders), **kwargs)

        plt.title(feature + ", " + attribute_name)

        save_name = feature + "_" + attribute_name + "_compare"

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()





    def plotCompareConfidenceIntervalFeature0d(self, feature, hardcopy=True,
                                               show=False, **kwargs):
        if feature not in self.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))


        width = 0.2
        distance = 0.5

        values = []
        xlabels = []
        xticks = []
        index = []
        pos = 0

        for compare in self.compare_folders:
            # xlabels.extend(["$P_{5}$", "$P_{95}$"])
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + width)
            index.extend([pos + 0.5*width])


            values.append(self.p_05_compare[compare][feature])
            # values.append(self.p_95_compare[compare][feature])

            pos += distance + 2*width

        prettyBar(values, index=index, xticks=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=2, color=0, label="$P_{5}$", **kwargs)



        values = []
        xlabels = []
        xticks = []
        index = []
        pos = 0

        for compare in self.compare_folders:
            # xlabels.extend(["$P_{5}$", "$P_{95}$"])
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + width)
            index.extend([pos + 1.5*width])


            values.append(self.p_05_compare[compare][feature])
            # values.append(self.p_95_compare[compare][feature])

            pos += distance + 2*width

        prettyBar(values, index=index, xticks=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=2, color=1, label="$P_{95}$", new_figure=False, **kwargs)


        plt.legend()


        plt.title(feature + ", 90 \% Confidence interval")

        save_name = feature + "_confidence-interval_compare"

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()



    # TODO create the avobe compare plots for the fractional difference also

    

    # def plotCompareAttributeFractionalDifference(self, feature="directComparison", attribute="E",
    #                                              reference_name="pc"):
    #     if feature not in self.features_1d:
    #         raise ValueError("%s is not a 1D feature" % (feature))
    #
    #     color = 0
    #     max_value = 0
    #     min_value = 10**10
    #     legend = []
    #     new_figure = True
    #
    #     value = getattr(self, attribute + "_compare")
    #
    #     compares = value.keys()
    #     compares.remove(reference_name)
    #
    #     for compare in compares:
    #         if compare[:3] == "mc_":
    #             legend.append("MC samples " + compare.split("_")[-1])
    #         else:
    #             legend.append(compare)
    #
    #         self.t = self.t_compare[compare]
    #         if attribute == "E":
    #             fractional_difference = abs(value[reference_name][feature] - value[compare][feature])/value[reference_name][feature]
    #             min_value, max_value = self.getMinMax(fractional_difference,
    #                                                   min_value, max_value)
    #
    #             self.E[feature] = fractional_difference
    #             self.plotMean(feature=feature, hardcopy=False, show=False,
    #                           new_figure=new_figure, color=color)
    #
    #         new_figure = False
    #         color += 2
    #
    #     if attribute == "E":
    #         title = "$\\frac{|{PC}_{mean} - MC_{mean}|}{PC_{mean}}$, " + feature
    #         save_name = feature + "mean_fractional-difference"
    #
    #     title(title)
    #
    #     plt.ylim([min_value*0.99, max_value*1.01])
    #     plt.legend(legend)
    #     # TODO add show and hardcopy options
    #     # plt.savefig(os.path.join(self.compare_output_dir_figures,
    #     #                          save_name + self.figureformat))
    #
    #     plt.show()



    def getMinMax(self, value, min_value, max_value):
        if value.max() > max_value:
            max_value = value.max()

        if value.min() < min_value:
            min_value = value.min()

        return min_value, max_value

















    def gif(self):
        self.logger.info("Creating gifs. May take a long time")

        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if not os.path.isdir(self.output_dir_gif):
            os.makedirs(self.output_dir_gif)

        plotting_order = {}
        for f in glob.glob(self.data_dir + "*"):
            if re.search("._\d", f):
                distribution, interval = f.split("/")[-1].split("_")

                if distribution in plotting_order:
                    plotting_order[distribution].append(interval)
                else:
                    plotting_order[distribution] = [interval]

        for distribution in plotting_order:
            plotting_order[distribution].sort()

        # Run through each distribution and plot everything for each distribution
        for distribution in plotting_order:
            if os.path.isdir(self.tmp_gif_output):
                shutil.rmtree(self.tmp_gif_output)
            os.makedirs(self.tmp_gif_output)

            max_data = {}
            min_data = {}
            filenames = []

            for interval in plotting_order[distribution]:
                foldername = distribution + "_" + interval

                for f in glob.glob(os.path.join(self.data_dir, foldername, "*")):
                    filename = f.split("/")[-1]

                    if filename not in max_data:
                        max_data[filename] = {}
                        min_data[filename] = {}

                    if filename not in filenames:
                        filenames.append(filename)

                    f = h5py.File(f, 'r')


                    for feature in f.keys():
                        if feature not in max_data[filename]:
                            max_data[filename][feature] = {}
                            min_data[filename][feature] = {}


                        for result in f[feature].keys():
                            if result == "t" or result == "sensitivity":
                                continue

                            max_value = f[feature][result][()].max()
                            min_value = f[feature][result][()].min()

                            if result in max_data[filename][feature]:
                                if max_value > max_data[filename][feature][result]:
                                    max_data[filename][feature][result] = max_value
                                if min_value < min_data[filename][feature][result]:
                                    min_data[filename][feature][result] = min_value
                            else:
                                max_data[filename][feature][result] = max_value
                                min_data[filename][feature][result] = min_value


            for interval in plotting_order[distribution]:
                foldername = distribution + "_" + interval

                for f in glob.glob(os.path.join(self.data_dir, foldername, "*")):
                    filename = f.split("/")[-1]
                    self.loadData(os.path.join(foldername, filename))

                    for feature in self.features_1d:
                        self.plotMean(feature=feature, hardcopy=False)
                        plt.ylim([min_data[filename][feature]["E"],
                                  max_data[filename][feature]["E"]])
                        save_name = filename + "_mean" + "_" + interval + self.figureformat
                        plt.title("Mean, " + feature + ", " + interval)
                        plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                        plt.ylim([min_data[filename][feature]["Var"],
                                  max_data[filename][feature]["Var"]])
                        save_name = filename + "_variance" + "_" + interval + self.figureformat
                        plt.title("Variance, " + feature + ", " + interval)
                        plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                        plt.close()

                        ax, ax2 = self.plotMeanAndVariance(feature=feature, hardcopy=False)

                        ax.set_ylim([min_data[filename][feature]["E"],
                                     max_data[filename][feature]["E"]])
                        ax2.set_ylim([min_data[filename][feature]["Var"],
                                      max_data[filename][feature]["Var"]])
                        save_name = filename + "_mean-variance" + "_" + interval + self.figureformat
                        ax.set_title("Mean and Variance, " + feature + ", " + interval)
                        plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                        plt.close()

                        self.plotConfidenceInterval(feature=feature, hardcopy=False)
                        plt.ylim([min(min_data[filename][feature]["p_95"],
                                      min_data[filename][feature]["p_05"]),
                                  max(max_data[filename][feature]["p_95"],
                                      max_data[filename][feature]["p_05"])])
                        save_name = filename + "_confidence-interval" + \
                            "_" + interval + self.figureformat
                        plt.title("Confidence interval, " + feature + ", " + interval)
                        plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                        plt.close()


                        # # PLot sensitivity each single plot
                        # if feature not in self.features_1d:
                        #     # TODO is this the right error to raise?
                        #     raise ValueError("%s is not a 1d feature" % (feature))
                        #
                        # if "sensitivity" not in self.f[feature].keys():
                        #     continue
                        #
                        # t = self.f["directComparison"]["t"][:]
                        # sensitivity = self.f[feature]["sensitivity"][:]
                        #
                        # parameter_names = self.uncertain_parameters
                        #
                        # for i in range(len(sensitivity)):
                        #     prettyPlot(t, sensitivity[i],
                        #                parameter_names[i] + " sensitivity", "time",
                        #                "sensitivity", i, True)
                        #     plt.title(parameter_names[i] + " sensitivity")
                        #     plt.ylim([0, 1.05])
                        #
                        #     save_name = filename + "_" + interval + "_sensitivity_" \
                        #           + parameter_names[i] + self.figureformat
                        #     plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                        #     plt.close()



                        self.plotSensitivityCombined(feature=feature, hardcopy=False)
                        save_name = filename + "_sensitivity" + "_" + interval + self.figureformat
                        plt.title("Sensitivity, " + feature + ", " + interval)
                        plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                        plt.close()




            # Create gif
            outputdir = os.path.join(self.output_dir_gif, distribution)
            # if os.path.isdir(outputdir):
            #     shutil.rmtree(outputdir)
            # os.makedirs(outputdir)
            if not os.path.isdir(outputdir):
                os.makedirs(outputdir)

            #
            plot_types = ["mean", "variance", "mean-variance", "confidence-interval", "sensitivity"]

            for fi in filenames:
                for plot_type in plot_types:
                    if "single-parameter" in f and plot_type == "sensitivity":
                        continue

                    final_name = os.path.join(outputdir, fi + "_" + plot_type)
                    filename = os.path.join(self.tmp_gif_output,
                                            "%s_%s_*%s" % (fi, plot_type, self.figureformat))

                    cmd = "convert -set delay 100 %s %s.gif" % (filename, final_name)

                    os.system(cmd)

            shutil.rmtree(self.tmp_gif_output)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot data")
    parser.add_argument("-d", "--data_dir", help="Directory the data is stored in")

    args = parser.parse_args()

    data_dir = "data/"
    output_dir_figures = "figures/"
    output_dir_gif = "gifs/"
    figureformat = ".png"

    if args.data_dir:
        data_dir = "%s/" % os.path.join(data_dir, args.data_dir)
        output_dir_figures = "%s/" % os.path.join(output_dir_figures, args.data_dir)
        output_dir_gif = "%s/" % os.path.join(output_dir_gif, args.data_dir)


    plot = PlotUncertainty(data_dir=data_dir,
                           output_dir_figures=output_dir_figures,
                           figureformat=figureformat,
                           output_dir_gif=output_dir_gif)

    # plot.plotAllData()
    plot.plotAllDataFromExploration()
    plot.gif()

    # sortByParameters(path=output_dir_figures, outputpath=output_dir_figures)
