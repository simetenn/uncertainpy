import os
import h5py
import sys
import shutil
import glob
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np

from uncertainpy.plotting.prettyPlot import prettyPlot, prettyBar
from uncertainpy.utils import sortByParameters

### TODO rewrite gif() to use less memory when creating GIF(Only load one dataset at the time)

### TODO Add feature plots to gif()

### TODO find a good way to find the directory where the data files are


class PlotUncertainty():
    def __init__(self,
                 data_dir="data/",
                 output_dir_figures="figures/",
                 output_dir_gif="gifs/",
                 figureformat=".png",
                 combined_features=True):

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


    def loadData(self, filename):
        self.filename = filename
        f = h5py.File(os.path.join(self.data_dir, self.filename), 'r')

        self.features_1d = []
        self.features_2d = []

        for feature in f.keys():
            if len(f[feature]["E"].shape) == 0:
                self.features_1d.append(feature)
            elif len(f[feature]["E"].shape) == 1:
                self.features_2d.append(feature)
            else:
                print "WARNING: No support for more than 1d and 2d plotting"

        self.full_output_dir_figures = os.path.join(self.output_dir_figures, filename)

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
                self.t = f[feature]["t"][()]
            # else:
            #     self.t[feature] = None

        self.uncertain_parameters = f.attrs["uncertain parameters"]
        self.loaded_flag = True


    def setData(self, t, U, E, Var, p_05, p_95, uncertain_parameters, sensitivity, foldername=None):

        self.t = t
        self.U = U
        self.E = E
        self.Var = Var
        self.p_05 = p_05
        self.p_95 = p_95
        self.sensitivity = sensitivity
        self.uncertain_parameters = uncertain_parameters

        self.features_1d = []
        self.features_2d = []

        for feature in self.E:
            if len(self.E[feature].shape) == 0:
                self.features_1d.append(feature)
            elif len(self.E[feature].shape) == 1:
                self.features_2d.append(feature)
            else:
                print "WARNING: No support for more than 1d and 2d plotting"


        if foldername is None:
            self.filename = ""
            self.full_output_dir_figures = self.output_dir_figures
        else:
            self.filename = foldername
            self.full_output_dir_figures = os.path.join(self.output_dir_figures, self.filename)

        if not os.path.isdir(self.full_output_dir_figures):
            os.makedirs(self.full_output_dir_figures)

        self.loaded_flag = True

    def plotMean(self, feature="direct_comparison", hardcopy=True, show=False):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        if feature not in self.features_2d:
            # TODO is this the right error to raise?
            raise ValueError("%s is not a 2D feature" % (feature))

        color1 = 0

        prettyPlot(self.t[feature], self.E[feature], "Mean, " + feature, "time", "voltage", color1)
        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_mean" + self.figureformat))
            plt.close()

        if show:
            plt.show()
            plt.close()



    def plotVariance(self, feature="direct_comparison", hardcopy=True, show=False):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        if feature not in self.features_2d:
            # TODO is this the right error to raise?
            raise ValueError("%s is not a 2D feature" % (feature))


        color2 = 8

        prettyPlot(self.t[feature], self.Var[feature], "Variance, " + feature,
                   "time", "voltage", color2)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     feature + "_variance" + self.figureformat))
            plt.close()

        if show:
            plt.show()
            plt.close()



    def plotMeanAndVariance(self, feature="direct_comparison", hardcopy=True, show=False):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        if feature not in self.features_2d:
            # TODO is this the right error to raise?
            raise ValueError("%s is not a 2D feature" % (feature))


        color1 = 0
        color2 = 8

        ax, tableau20 = prettyPlot(self.t[feature], self.E[feature],
                                   "Mean and variance, " + feature, "time", "voltage, mean", color1)
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=tableau20[color2], labelcolor=tableau20[color2], labelsize=14)
        ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
        ax.spines["right"].set_edgecolor(tableau20[color2])

        ax2.set_xlim([min(self.t[feature]), max(self.t[feature])])
        ax2.set_ylim([min(self.Var[feature]), max(self.Var[feature])])

        ax2.plot(self.t[feature], self.Var[feature],
                 color=tableau20[color2], linewidth=2, antialiased=True)

        ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
        ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
        ax.spines["left"].set_edgecolor(tableau20[color1])
        # plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                        feature + "_variance-mean" + self.figureformat))
            plt.close()

        if show:
            plt.show()
            plt.close()

        if not show or not hardcopy:
            return ax, ax2



    def plotConfidenceInterval(self, feature="direct_comparison", hardcopy=True, show=False):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        if feature not in self.features_2d:
            # TODO is this the right error to raise?
            raise ValueError("%s is not a 2D feature" % (feature))


        ax, color = prettyPlot(self.t[feature], self.E[feature],
                               xlabel="time", ylabel="voltage", color=0)
        plt.fill_between(self.t[feature], self.p_05[feature], self.p_95[feature],
                         alpha=0.2, facecolor=color[8])
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
            plt.close()

        if show:
            plt.show()
            plt.close()


    def plotSensitivity(self, feature="direct_comparison", hardcopy=True, show=False):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        if feature not in self.features_2d:
            # TODO is this the right error to raise?
            raise ValueError("%s is not a 2D feature" % (feature))

        if feature not in self.sensitivity or self.sensitivity[feature] is None:
            return

        parameter_names = self.uncertain_parameters

        for i in range(len(self.sensitivity[feature])):
            prettyPlot(self.t[feature], self.sensitivity[feature][i],
                       parameter_names[i] + " sensitivity", "time",
                       "sensitivity", i, True)
            plt.ylim([0, 1.05])

            if hardcopy:
                plt.savefig(os.path.join(self.full_output_dir_figures,
                                         "Sensitivity_" + feature + "_" + parameter_names[i] + self.figureformat))
                plt.close()

            if show:
                plt.show()
                plt.close()



    def plotSensitivityCombined(self, feature="direct_comparison", hardcopy=True, show=False):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        if feature not in self.features_2d:
            # TODO is this the right error to raise?
            raise ValueError("%s is not a 2D feature" % (feature))

        if feature not in self.sensitivity or self.sensitivity[feature] is None:
            return

        parameter_names = self.uncertain_parameters

        for i in range(len(self.sensitivity[feature])):
            prettyPlot(self.t[feature], self.sensitivity[feature][i], "sensitivity", "time",
                       "Sensitivity, " + feature, i, False)

        plt.ylim([0, 1.05])
        plt.xlim([self.t[feature][0], 1.3*self.t[feature][-1]])
        plt.legend(parameter_names)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     "sensitivity" + self.figureformat))
            plt.close()

        if show:
            plt.show()
            plt.close()


    def plot2dFeatures(self):
        for feature in self.features_2d:
            self.plotMean(feature=feature)
            self.plotVariance(feature=feature)
            self.plotMeanAndVariance(feature=feature)
            self.plotConfidenceInterval(feature=feature)
            self.plotSensitivity(feature=feature)
            self.plotSensitivityCombined(feature=feature)



    def plot1dFeaturesCombined(self, index=0, max_legend_size=5):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)


        if len(self.uncertain_parameters) > 8:
            self.features_in_combined_plot = 2

        if self.features_in_combined_plot + index < len(self.features_1d):
            self.plot1dFeaturesCombined(index + self.features_in_combined_plot)
            feature_names = self.features_1d[index:self.features_in_combined_plot + index]
        else:
            feature_names = self.features_1d[index:]

        if len(feature_names) == 0:
            return

        axis_grey = (0.5, 0.5, 0.5)
        titlesize = 18
        fontsize = 16
        labelsize = 14

        if len(self.features_1d) > max_legend_size:
            legend_size = max_legend_size
        else:
            legend_size = len(self.uncertain_parameters)

        legend_width = np.ceil(len(self.features_1d)/float(max_legend_size))

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
        fig, ax_all = plt.subplots(1, len(feature_names))

        if len(feature_names) == 1:
            ax_all = [ax_all]

        for feature_name in feature_names:
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


            # if "sensitivity" in self.f[feature_name].keys():
            #     sensitivity = self.f[feature_name]["sensitivity"][:]

            ax.bar(pos, self.E[feature_name],
                   width=width, align='center', color=tableau20[1], linewidth=0)
            xticks.append(pos - 0.5*width)
            xticklabels.append("Mean")
            pos += width


            ax.bar(pos, self.Var[feature_name],
                   width=width, align='center', color=tableau20[0], linewidth=0)
            xticks.append(pos - 0.5*width)
            xticklabels.append("Variance")
            pos += distance

            ax.bar(pos, self.p_05[feature_name],
                   width=width, align='center', color=tableau20[3], linewidth=0)
            ax.bar(pos + width, self.p_95[feature_name],
                   width=width, align='center', color=tableau20[2], linewidth=0)
            xticks += [pos - 0.5*width, pos + 0.5*width]
            xticklabels += ["$P_5$", "$P_{95}$"]


            if feature_name in self.sensitivity and self.sensitivity[feature_name] is not None:
                pos += distance + width

                i = 0
                legend_bars = []

                for parameter in self.uncertain_parameters:
                    # TODO is abs(sensitivity) a problem in the plot?
                    legend_bars.append(ax2.bar(pos, abs(self.sensitivity[feature_name][i]),
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

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45,
                               horizontalalignment="left")
            ax.set_title(feature_name)

            ax_i += 1


        ax_all[0].set_ylabel('Feature value', fontsize=fontsize)
        ax2.set_ylabel('Sensitivity', fontsize=fontsize, color=tableau20[4])

        if feature_name in self.sensitivity and self.sensitivity[feature_name] is not None:
                # Put a legend above current axis
            if len(feature_names) == 1:
                location = (0.5, 1.03 + legend_width*0.095)
                loc = "upper center"
            elif len(feature_names) == 2:
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

        save_name = self.filename + \
            ("_features_%d" % (index/self.features_in_combined_plot)) + self.figureformat
        plt.savefig(os.path.join(self.full_output_dir_figures, save_name))
        plt.close()


    def plot1dFeatures(self):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        for feature_name in self.features_1d:
            self.plot1dFeature(feature_name)
            save_name = "%s_%s" % (self.filename, feature_name) + self.figureformat
            plt.savefig(os.path.join(self.full_output_dir_figures, save_name))
            plt.close()



    # TODO not finhised, missing correct label placement
    def plot1dFeature(self, feature_name, max_legend_size=5):
        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

        if feature_name not in self.features_1d:
            # TODO is this the right error to raise?
            raise ValueError("%s is not a 1D feature" % (feature_name))


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
        ax, tableau20 = prettyBar(self.E[feature_name], start_color=1)
        ax.spines["right"].set_edgecolor(axis_grey)

        ax.set_ylabel(feature_name, fontsize=labelsize)
        pos += width

        ax.bar(pos, self.Var[feature_name],
               width=width, align='center', color=tableau20[0], linewidth=0)
        xticks.append(pos)
        xticklabels += ["Variance"]
        pos += distance


        ax.bar(pos, self.p_05[feature_name],
               width=width, align='center', color=tableau20[3], linewidth=0)
        ax.bar(pos + width, self.p_95[feature_name],
               width=width, align='center', color=tableau20[2], linewidth=0)
        xticks += [pos, pos + width]
        xticklabels += ["$P_5$", "$P_{95}$"]
        pos += distance + width

        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=axis_grey, labelcolor=tableau20[4], labelsize=labelsize)
        ax2.set_ylabel('Sensitivity', fontsize=fontsize, color=tableau20[4])
        ax2.set_ylim([0, 1.05])

        if self.sensitivity[feature_name] is not None:
            i = 0
            legend_bars = []
            for parameter in self.uncertain_parameters:
                legend_bars.append(ax2.bar(pos, self.sensitivity[feature_name][i], width=width,
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
        plt.suptitle(self.filename + ", " + feature_name, fontsize=titlesize)

        return ax, tableau20, pos



    def plotAllDataInFolder(self):
        print "Plotting all data"

        for f in glob.glob(os.path.join(self.data_dir, "*")):
            self.loadData(f.split("/")[-1])

            self.plotAllData()


    def plotAllData(self):
        self.plot2dFeatures()

        if self.combined_features:
            self.plot1dFeaturesCombined()
        else:
            self.plot1dFeatures()




    def plotAllDataFromExploration(self):
        print "Plotting all data"

        original_data_dir = self.data_dir
        original_output_dir_figures = self.output_dir_figures

        for folder in glob.glob(os.path.join(self.data_dir, "*")):
            self.data_dir = os.path.join(original_data_dir, folder.split("/")[-1])
            self.output_dir_figures = os.path.join(original_output_dir_figures,
                                                   folder.split("/")[-1])

            for filename in glob.glob(os.path.join(folder, "*")):

                self.loadData(filename.split("/")[-1])

                self.plot2dFeatures()

                if self.combined_features:
                    self.plot1dFeaturesCombined()
                else:
                    self.plot1dFeatures()

        self.data_dir = original_data_dir
        self.output_dir_figures = original_output_dir_figures


    def gif(self):
        print "Creating gifs..."

        if not self.loaded_flag:
            print "Datafile must be loaded"
            sys.exit(1)

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

                    for feature in self.features_2d:
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
                        # if feature not in self.features_2d:
                        #     # TODO is this the right error to raise?
                        #     raise ValueError("%s is not a 2D feature" % (feature))
                        #
                        # if "sensitivity" not in self.f[feature].keys():
                        #     continue
                        #
                        # t = self.f["direct_comparison"]["t"][:]
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
