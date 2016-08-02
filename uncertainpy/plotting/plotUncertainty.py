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
from uncertainpy.plotting.prettyPlot import *
from uncertainpy.utils import create_logger

# TODO rewrite gif() to use less memory when creating GIF(Only load one dataset at the time)

# TODO Add feature plots to gif()

# TODO find a good way to find the directory where the data files are

# TODO test out if seaborn is a better package to use for plottintg

# TODO move load() to it's own class


class PlotUncertainty():
    def __init__(self,
                 data_dir="data/",
                 output_dir_figures="figures/",
                 output_dir_gif="gifs/",
                 figureformat=".png",
                 combined_features=True,
                 verbose_level="info",
                 verbose_filename=None,):

        self.data_dir = data_dir
        self.output_dir_figures = output_dir_figures
        self.output_dir_gif = output_dir_gif
        self.figureformat = figureformat
        self.combined_features = combined_features
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
                 new_figure=True, color=0):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        prettyPlot(self.t[feature], self.E[feature],
                   "Mean, " + feature, "time", "voltage", color=color,
                   new_figure=False)


        min_t = self.t[feature].min()
        max_t = self.t[feature].max()

        min_E = self.E[feature].min()
        max_E = self.E[feature].max()

        plt.xlim(min_t, max_t)
        plt.ylim(min_E, max_E)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_mean" + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()

        return min_t, max_t, min_E, max_E



    def plotVariance(self, feature="directComparison",
                     color=8, new_figure=True, hardcopy=True, show=False):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        prettyPlot(self.t[feature], self.Var[feature], "Variance, " + feature,
                   "time", "voltage", color=color, new_figure=new_figure)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_variance" + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()



    def plotMeanAndVariance(self, feature="directComparison", color=0, new_figure=True, hardcopy=True, show=False):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        tableau20 = colormap()

        ax= prettyPlot(self.t[feature], self.E[feature],
                       "Mean and variance, " + feature, "time", "voltage, mean", color=color, new_figure=new_figure)
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=tableau20[color+2], labelcolor=tableau20[color+2], labelsize=14)
        ax2.set_ylabel('voltage, variance', color=tableau20[color+2], fontsize=16)
        ax.spines["right"].set_edgecolor(tableau20[color+2])

        ax2.set_xlim([min(self.t[feature]), max(self.t[feature])])
        ax2.set_ylim([min(self.Var[feature]), max(self.Var[feature])])

        ax2.plot(self.t[feature], self.Var[feature],
                 color=tableau20[color+2], linewidth=2, antialiased=True)

        ax.tick_params(axis="y", color=tableau20[color], labelcolor=tableau20[color])
        ax.set_ylabel('voltage, mean', color=tableau20[color], fontsize=16)
        ax.spines["left"].set_edgecolor(tableau20[color])
        # plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                        feature + "_mean-variance" + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()

        if not show or not hardcopy:
            return ax, ax2



    def plotConfidenceInterval(self, feature="directComparison", hardcopy=True, show=False):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        tableau20 = colormap()

        ax = prettyPlot(self.t[feature], self.E[feature],
                        xlabel="time", ylabel="voltage", color=0)
        plt.fill_between(self.t[feature], self.p_05[feature], self.p_95[feature],
                         alpha=0.2, facecolor=tableau20[8])
        prettyPlot(self.t[feature], self.p_95[feature], color=8, new_figure=False)
        prettyPlot(self.t[feature], self.p_05[feature], color=9, new_figure=False)
        prettyPlot(self.t[feature], self.E[feature],
                   title="Confidence interval, " + feature, new_figure=False)

        plt.ylim([min([min(self.p_95[feature]), min(self.p_05[feature]), min(self.E[feature])]),
                  max([max(self.p_95[feature]), max(self.p_05[feature]), max(self.E[feature])])])

        plt.legend(["Mean", "$P_{95}$", "$P_{5}$"])

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_confidence-interval" + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()


    def plotSensitivity(self, feature="directComparison", hardcopy=True, show=False):
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
                       color=i, new_figure=True)
            plt.ylim([0, 1.05])

            if hardcopy:
                plt.savefig(os.path.join(self.full_output_dir_figures,
                                         feature + "_sensitivity_" + parameter_names[i] + self.figureformat))
                if not show:
                    plt.close()

            if show:
                plt.show()



    def plotSensitivityCombined(self, feature="directComparison",
                                hardcopy=True, show=False):
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
                       color=i, new_figure=False)

        plt.ylim([0, 1.05])
        plt.xlim([self.t[feature][0], 1.3*self.t[feature][-1]])
        plt.legend(parameter_names)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_sensitivity" + self.figureformat))
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

        axis_grey = (0.5, 0.5, 0.5)
        titlesize = 18
        fontsize = 16
        labelsize = 14

        width = 0.2
        distance = 0.5

        pos = 0
        xticks = [pos]
        xticklabels = ["mean"]
        ax, tableau20 = prettyBar(self.E[feature], start_color=1)
        ax.spines["right"].set_edgecolor(axis_grey)

        ax.set_ylabel(feature, fontsize=labelsize)
        pos += width

        ax.bar(pos, self.Var[feature],
               width=width, align='center', color=tableau20[0], linewidth=0)
        xticks.append(pos)
        xticklabels += ["Variance"]
        pos += distance


        ax.bar(pos, self.p_05[feature],
               width=width, align='center', color=tableau20[3], linewidth=0)
        ax.bar(pos + width, self.p_95[feature],
               width=width, align='center', color=tableau20[2], linewidth=0)
        xticks += [pos, pos + width]
        xticklabels += ["$P_5$", "$P_{95}$"]
        pos += distance + width

        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=axis_grey, labelcolor=tableau20[4], labelsize=labelsize)
        ax2.set_ylabel('Sensitivity', fontsize=fontsize, color=tableau20[4])
        ax2.set_ylim([0, 1.05])

        if self.sensitivity[feature] is not None:
            i = 0
            legend_bars = []
            for parameter in self.uncertain_parameters:
                legend_bars.append(ax2.bar(pos, self.sensitivity[feature][i], width=width,
                                           align='center', color=tableau20[4+i], linewidth=0))

                i += 1
                pos += width

            xticks.append(pos - width*i/2.)
            xticklabels.append("Sensitivity")

            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0,
            #                 box.width, box.height*(0.91 + legend_width*0.053)])
            # ax2.set_position([box.x0, box.y0,
            #                   box.width, box.height*(0.91 + legend_width*0.053

            location = (0.5, 1.01 + legend_width*0.095)
            lgd = plt.legend(legend_bars, self.uncertain_parameters,
                             loc='upper center', bbox_to_anchor=location,
                             fancybox=False, shadow=False, ncol=legend_size)
            lgd.get_frame().set_edgecolor(axis_grey)

            fig = plt.gcf()
            fig.subplots_adjust(top=(0.91 - legend_width*0.053))

        # else:
        #     ax.bar(pos, sensitivity, width=width, align='center', color=tableau20[4], linewidth=0)
        #     xticks.append(pos)
        #     xticklabels.append(self.filename)


        pos += 3*distance
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45)
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


        return ax, tableau20, pos



    def plot0dFeatures(self, hardcopy=True, show=False):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        for feature in self.features_0d:
            self.plot0dFeature(feature, hardcopy=hardcopy, show=show)



    def plot0dFeaturesCombined(self, index=0, max_legend_size=5,
                               hardcopy=True, show=False):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
        #Options
        params = {'text.usetex': True,
                  'font.family': 'lmodern',
                  'axes.grid': False,
                  'grid.color': 'white',
                  'grid.linewidth': 1.3,
                  'grid.linestyle': '-',
                  'axes.facecolor': '0.95',
                  'legend.fontsize': 16}

        plt.rcParams.update(params)


        if len(self.uncertain_parameters) > 8:
            self.features_in_combined_plot = 2

        if self.features_in_combined_plot + index < len(self.features_0d):
            self.plot0dFeaturesCombined(index + self.features_in_combined_plot)
            features = self.features_0d[index:self.features_in_combined_plot + index]
        else:
            features = self.features_0d[index:]

        if len(features) == 0:
            return

        axis_grey = (0.5, 0.5, 0.5)
        titlesize = 18
        fontsize = 16
        labelsize = 14

        if len(self.features_0d) > max_legend_size:
            legend_size = max_legend_size
        else:
            legend_size = len(self.uncertain_parameters)

        legend_width = np.ceil(len(self.features_0d)/float(max_legend_size))

        width = 0.2
        distance = 0.5

        tableau20 = [(31, 119, 180), (14, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        for i in range(len(tableau20)):
            r, g, b = tableau20[i]
            tableau20[i] = (r / 255., g / 255., b / 255.)

        ax_i = 0
        fig, ax_all = plt.subplots(1, len(features))

        if len(features) == 1:
            ax_all = [ax_all]

        for feature in features:
            pos = 0
            xticks = []
            xticklabels = []
            ax = ax_all[ax_i]

            ax.spines["top"].set_edgecolor("None")
            ax.spines["bottom"].set_edgecolor(axis_grey)
            ax.spines["right"].set_edgecolor(axis_grey)
            ax.spines["left"].set_edgecolor(axis_grey)
            #
            ax.tick_params(axis="x", which="both", bottom="on", top="off",
                           labelbottom="on", color=axis_grey, labelcolor="black",
                           labelsize=labelsize)
            ax.tick_params(axis="y", which="both", right="off", left="off",
                           labelleft="off", color=axis_grey, labelcolor="black",
                           labelsize=labelsize)

            ax2 = ax.twinx()
            ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                            color=axis_grey, labelcolor=tableau20[4], labelsize=labelsize)

            ax2.set_ylim([0, 1.05])


            # if "sensitivity" in self.f[feature].keys():
            #     sensitivity = self.f[feature]["sensitivity"][:]

            ax.bar(pos, self.E[feature],
                   width=width, align='center', color=tableau20[1], linewidth=0)
            xticks.append(pos - 0.5*width)
            xticklabels.append("Mean")
            pos += width


            ax.bar(pos, self.Var[feature],
                   width=width, align='center', color=tableau20[0], linewidth=0)
            xticks.append(pos - 0.5*width)
            xticklabels.append("Variance")
            pos += distance

            ax.bar(pos, self.p_05[feature],
                   width=width, align='center', color=tableau20[3], linewidth=0)
            ax.bar(pos + width, self.p_95[feature],
                   width=width, align='center', color=tableau20[2], linewidth=0)
            xticks += [pos - 0.5*width, pos + 0.5*width]
            xticklabels += ["$P_5$", "$P_{95}$"]

            if feature in self.sensitivity and self.sensitivity[feature] is not None:
                pos += distance + width

                i = 0
                legend_bars = []

                for parameter in self.uncertain_parameters:
                    # TODO is abs(sensitivity) a problem in the plot?
                    legend_bars.append(ax2.bar(pos, abs(self.sensitivity[feature][i]),
                                               width=width, align='center', color=tableau20[4+i],
                                               linewidth=0))

                    i += 1
                    pos += width

                xticks.append(pos - width*i/2.)
                xticklabels.append("Sensitivity")

                #
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0,
                #                 box.width, box.height*(1 - legend_width*0.1)])
                # ax2.set_position([box.x0, box.y0,
                #                   box.width, box.height*(1 - legend_width*0.1)])
            # else:
                # TODO is abs(sensitivity) a problem in the plot?
                # ax2.bar(pos, abs(sensitivity), width=width, align='center', color=tableau20[4],
                #  linewidth=0)
                # xticks.append(pos + distance)
                # xticklabels.append("")

            xticks += [pos + distance]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45,
                               horizontalalignment="left")
            ax.set_title(feature)

            ax_i += 1


        ax_all[0].set_ylabel('Feature value', fontsize=fontsize)
        ax2.set_ylabel('Sensitivity', fontsize=fontsize, color=tableau20[4])

        if feature in self.sensitivity and self.sensitivity[feature] is not None:
                # Put a legend above current axis
            if len(features) == 1:
                location = (0.5, 1.03 + legend_width*0.095)
                loc = "upper center"
            elif len(features) == 2:
                location = (0, 1.03 + legend_width*0.095)
                loc = "upper center"
            else:
                location = (0.15, (1.03 + legend_width*0.095))
                loc = "upper right"


            lgd = ax.legend(legend_bars, self.uncertain_parameters, loc=loc,
                            bbox_to_anchor=location, fancybox=False,
                            shadow=False, ncol=legend_size)
            lgd.get_frame().set_edgecolor(axis_grey)

            # plt.tight_layout()
            fig.subplots_adjust(top=(0.91 - legend_width*0.053), wspace=0.5)
        else:
            fig.subplots_adjust(wspace=0.5)

        plt.suptitle(self.filename, fontsize=titlesize)

        save_name = "combined_features_%d" % (index/self.features_in_combined_plot) + self.figureformat


        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures, save_name))

            if not show:
                plt.close()

        if show:
            plt.show()



    def plotAllDataInFolder(self):
        self.logger.info("Plotting all data in folder")

        for f in glob.glob(os.path.join(self.data_dir, "*")):
            self.loadData(f.split(os.path.sep)[-1])

            self.plotAllData()


    def plotAllData(self):
        self.plot1dFeatures()

        if self.combined_features:
            self.plot0dFeaturesCombined()
        else:
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

                if self.combined_features:
                    self.plot0dFeaturesCombined()
                else:
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
                                                             compare_folders))]

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



        # color = 0
        # max_value = 0
        # min_value = 10**10
        # legend = []
        # new_figure = True
        #
        #
        # compares = self.E_compare.keys()
        # compares.remove(self.reference_name)

        # for compare in compares:
        #     min_value, max_value = self.getMinMax(value[compare][feature],
        #                                           min_value, max_value)
        #
        #     if compare[:3] == "mc_":
        #         legend.append("MC samples " + compare.split("_")[-1])
        #     else:
        #         legend.append(compare)
        #
        #
        #     if attribute == "E":
        #         self.E = value[compare]
        #         self.plotMean(feature=feature, hardcopy=False, show=False,
        #                       new_figure=new_figure, color=color)
        #
        #     new_figure = False
        #     color += 2
        #
        #
        # min_value, max_value = self.getMinMax(value[self.reference_name][feature],
        #                                       min_value, max_value)
        # legend.append("PC")
        # if attribute == "E":
        #     self.E = value[self.reference_name]
        #     self.plotMean(feature=feature, hardcopy=False, show=False,
        #                   new_figure=new_figure, color=color)
        #     # TODO avoid using this function
        #     setTitle("Compare mean, " + feature)
        #     save_name = "MC-PC_comparison_" + feature
        #
        #
        # plt.ylim([min_value*0.99, max_value*1.01])
        # plt.legend(legend)
        # plt.show()
        # plt.savefig(os.path.join(output_dir_compare, save_name + self.figureformat))




    def plotCompareMean(self, feature="directComparison",
                        hardcopy=True, show=False):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_values = []
        min_values = []
        legend = []
        new_figure = True

        for compare in self.compare_folders:
            min_values.append(self.E_compare[compare][feature].min())
            max_values.append(self.E_compare[compare][feature].max())


            if compare[:2] == "mc":
                legend.append("MC samples " + compare.split("_")[-1])
            else:
                legend.append(compare)

            self.t = self.t_compare[compare]
            self.E = self.E_compare[compare]

            self.plotMean(feature=feature, hardcopy=False, show=False,
                          new_figure=new_figure, color=color)

            new_figure = False
            color += 2

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



    def plotCompareVariance(self, feature="directComparison",
                            hardcopy=True, show=False):
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


            if compare[:2] == "mc":
                legend.append("MC samples " + compare.split("_")[-1])
            else:
                legend.append(compare)

            self.t = self.t_compare[compare]
            self.Var = self.Var_compare[compare]

            self.plotVariance(feature=feature, hardcopy=False, show=False,
                              new_figure=new_figure, color=color)

            new_figure = False
            color += 2

        save_name = feature + "_variance_compare"

        plt.ylim([min(min_values)*0.99, max(max_values)*1.01])
        plt.legend(legend)

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()



    def plotCompareMeanAndVariance(self, feature="directComparison",
                                   hardcopy=True, show=False):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_values_E = []
        min_values_E = []
        max_values_Var = []
        min_values_Var = []

        legend_E = []
        legend_Var = []
        new_figure = True
        ax2 = None


        for compare in self.compare_folders:
            min_values_E.append(self.E_compare[compare][feature].min())
            max_values_E.append(self.E_compare[compare][feature].max())
            min_values_Var.append(self.Var_compare[compare][feature].min())
            max_values_Var.append(self.Var_compare[compare][feature].max())


            if compare[:2] == "mc":
                nr_mc_samples = compare.split("_")[-1]
                legend_E.append("MC samples " + nr_mc_samples)
                legend_Var.append("MC samples " + nr_mc_samples)
            else:
                legend_E.append(compare)
                legend_Var.append(compare)

            self.t = self.t_compare[compare]
            self.E = self.E_compare[compare]
            self.Var = self.Var_compare[compare]

            if new_figure:
                ax, tableau20 = prettyPlot(self.t[feature], self.E[feature],
                                           "Mean and variance, " + feature,
                                           "time",
                                           "voltage, mean",
                                           color=color,
                                           new_figure=new_figure,
                                           grid=False)
                ax2 = ax.twinx()
                ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                                color=tableau20[color+2], labelcolor=tableau20[color+2], labelsize=14)
                ax2.set_ylabel('voltage, variance', color=tableau20[color+2], fontsize=16)

                ax.tick_params(axis="y", color=tableau20[color], labelcolor=tableau20[color])
                ax.set_ylabel('voltage, mean', color=tableau20[color], fontsize=16)
                ax.spines["left"].set_edgecolor(tableau20[color])

                ax.spines["right"].set_edgecolor(tableau20[color+2])


            else:
                ax.plot(self.t[feature], self.E[feature],
                        color=tableau20[color], linewidth=2, antialiased=True,
                        zorder=3)

            ax2.plot(self.t[feature], self.Var[feature],
                     color=tableau20[color+2], linewidth=2, antialiased=True,
                     zorder=3)

            new_figure = False
            color += 4

        save_name = feature + "_mean-variance_compare"

        legend1 = ax.legend(legend_E, loc=2, title="Mean", fontsize=16)
        legend2 = ax2.legend(legend_Var, title="Variance", fontsize=16)

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
                                      hardcopy=True, show=False):

        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_values = []
        min_values = []
        legend = []
        new_figure = True


        color_table = colormap()

        for compare in self.compare_folders:

            if compare[:2] == "mc":
                legend.append("MC samples " + compare.split("_")[-1])
            else:
                legend.append(compare)

            self.t = self.t_compare[compare]
            self.E = self.E_compare[compare]
            self.p_05 = self.p_05_compare[compare]
            self.p_95 = self.p_95_compare[compare]

            if new_figure:
                ax = prettyPlot(self.t[feature], self.E[feature],
                                xlabel="time", ylabel="voltage",
                                color=color, new_figure=new_figure,
                                grid=False)


            ax.fill_between(self.t[feature], self.p_05[feature], self.p_95[feature],
                            alpha=0.2, facecolor=color_table[color+2])

            ax.plot(self.t[feature], self.E[feature],
                    color=color_table[color], linewidth=2, antialiased=True)
            ax.plot(self.t[feature], self.p_05[feature],
                    color=color_table[color+2], linewidth=2, antialiased=True)
            ax.plot(self.t[feature], self.p_95[feature],
                    color=color_table[color+3], linewidth=2, antialiased=True)


            min_values.append(min([min(self.p_95[feature]),
                                   min(self.p_05[feature]),
                                   min(self.E[feature])]))
            max_values.append(max([max(self.p_95[feature]),
                                   max(self.p_05[feature]),
                                   max(self.E[feature])]))

            legend.extend(["Mean", "$P_{95}$", "$P_{5}$"])



            new_figure = False
            color += 4
            #
            # plt.show()

        save_name = feature + "_confidence-interval_compare"

        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])
        plt.legend(legend, ncol=len(self.compare_folders))

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()






    # def plotCompare(self, plot_function, values=[], name=None, feature="directComparison",
    #                 hardcopy=True, show=False):
    #     if feature not in self.features_1d:
    #         raise ValueError("%s is not a 1D feature" % (feature))
    #
    #     allowed_plot_functions = ["plotMean", "plotVariance",
    #                               "plotMeanAndVariance", "plotConfidenceInterval"]
    #
    #     if not hasattr(plot_function, "__call__"):
    #         raise ValueError("plot_function must be callable")
    #
    #     if plot_function not in allowed_plot_functions:
    #         raise ValueError("Not allowed plot function, Working functions are: " + ', '.join(allowed_plot_functions))
    #
    #     if name is None:
    #         name = plot_function.__name__
    #
    #     color = 0
    #     legend = []
    #     new_figure = True
    #     min_xs = []
    #     max_xs = []
    #     min_ys = []
    #     max_ys = []
    #
    #
    #     for compare in self.compare_folders:
    #
    #         if compare[:2] == "mc":
    #             legend.append("MC samples " + compare.split("_")[-1])
    #         else:
    #             legend.append(compare)
    #
    #         self.t = self.t_compare[compare]
    #         #self.U = self.U_compare[compare]
    #         self.E = self.E_compare[compare]
    #         self.Var = self.Var_compare[compare]
    #         self.p_05 = self.p_05_compare[compare]
    #         self.p_95 = self.p_95_compare[compare]
    #         self.sensitivity = self.sensitivity_compare[compare]
    #
    #         min_x, max_x, min_y, max_y = plot_function(feature=feature,
    #                                                    hardcopy=False,
    #                                                    show=False,
    #                                                    new_figure=new_figure,
    #                                                    color=color)
    #         min_xs.append(min_x)
    #         max_xs.append(max_x)
    #         min_ys.append(min_y)
    #         max_ys.append(max_y)
    #
    #
    #         new_figure = False
    #         color += 2
    #
    #
    #
    #     save_name = feature + "_" + name
    #
    #     plt.xlim([min(min_xs)*0.99, max(max_xs)*1.01])
    #     plt.xlim([min(min_xs)*0.99, max(max_xs)*1.01])
    #     plt.legend(legend)
    #
    #     if hardcopy:
    #         plt.savefig(os.path.join(self.compare_output_dir_figures,
    #                                  save_name + self.figureformat))
    #         if not show:
    #             plt.close()
    #
    #     if show:
    #         plt.show()



    def plotCompareAttribute(self, feature="directComparison", attribute="E"):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_value = 0
        min_value = 10**10
        legend = []
        new_figure = True

        value = getattr(self, attribute + "_compare")

        for compare in self.compare_folders:
            min_value, max_value = self.getMinMax(value[compare][feature],
                                                  min_value, max_value)

            if compare[:2] == "mc":
                legend.append("MC samples " + compare.split("_")[-1])
            else:
                legend.append(compare)

            self.t = self.t_compare[compare]
            if attribute == "E":
                self.E = value[compare]
                self.plotMean(feature=feature, hardcopy=False, show=False,
                              new_figure=new_figure, color=color)
            elif attribute == "Var":
                self.Var = value[compare]
                self.plotVariance(feature=feature, hardcopy=False, show=False,
                                  new_figure=new_figure, color=color)
            elif attribute == "E_Var":
                # TODO working here: trying to figure out how to handle plotmeanandvariance()
                # Possible best solution, make own functions for comparing mean, variacne, mean and avriance and so on
                "he"
            else:
                raise ValueError("Unknown attribute {}".format(attribute))
            new_figure = False
            color += 2


        if attribute == "E":
            save_name = feature + "_mean"

        plt.ylim([min_value*0.99, max_value*1.01])
        plt.legend(legend)
        # TODO add show and hardcopy options
        plt.savefig(os.path.join(self.compare_output_dir_figures,
                                 save_name + self.figureformat))


    def plotCompareAttributeFractionalDifference(self, feature="directComparison", attribute="E",
                                                 reference_name="pc"):
        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_value = 0
        min_value = 10**10
        legend = []
        new_figure = True

        value = getattr(self, attribute + "_compare")

        compares = value.keys()
        compares.remove(reference_name)

        for compare in compares:
            if compare[:3] == "mc_":
                legend.append("MC samples " + compare.split("_")[-1])
            else:
                legend.append(compare)

            self.t = self.t_compare[compare]
            if attribute == "E":
                fractional_difference = abs(value[reference_name][feature] - value[compare][feature])/value[reference_name][feature]
                min_value, max_value = self.getMinMax(fractional_difference,
                                                      min_value, max_value)

                self.E[feature] = fractional_difference
                self.plotMean(feature=feature, hardcopy=False, show=False,
                              new_figure=new_figure, color=color)

            new_figure = False
            color += 2

        if attribute == "E":
            title = "$\\frac{|{PC}_{mean} - MC_{mean}|}{PC_{mean}}$, " + feature
            save_name = feature + "mean_fractional-difference"

        title(title)

        plt.ylim([min_value*0.99, max_value*1.01])
        plt.legend(legend)
        # TODO add show and hardcopy options
        # plt.savefig(os.path.join(self.compare_output_dir_figures,
        #                          save_name + self.figureformat))

        plt.show()



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
                        plt.close()

                        self.plotVariance(feature=feature, hardcopy=False)
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
