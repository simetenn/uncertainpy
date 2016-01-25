import os
import h5py
import sys
import shutil
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np

from prettyPlot import prettyPlot, prettyBar
from uncertainpy.utils import sortByParameters

### TODO rewrite gif() to use less memory when creating GIF(Only load one dataset at the time)

### TODO refactor gif() to be less complex and use more functions

### TODO Add feature plots to gif()

### TODO find a good way to find the directory where the data files are

class PlotUncertainty():
    def __init__(self,
                 data_dir="data/",
                 output_figures_dir="figures/",
                 output_gif_dir="gifs/",
                 figureformat=".png"):

        self.data_dir = data_dir
        self.output_figures_dir = output_figures_dir
        self.output_gif_dir = output_gif_dir
        self.figureformat = figureformat
        self.f = None

        self.tmp_gif_output = ".tmp_gif_output/"
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.features_in_combined_plot = 3

    def loadData(self, filename):
        self.filename = filename
        self.f = h5py.File(os.path.join(self.data_dir, self.filename), 'r')

        self.features_1d = []
        self.features_2d = []

        for feature in self.f.keys():
            if len(self.f[feature]["E"].shape) == 0:
                self.features_1d.append(feature)
            elif len(self.f[feature]["E"].shape) == 1:
                self.features_2d.append(feature)
            else:
                print "WARNING: No support for more than 1d and 2d plotting"

        self.full_output_figures_dir = os.path.join(self.output_figures_dir, filename)

        if os.path.isdir(self.full_output_figures_dir):
            shutil.rmtree(self.full_output_figures_dir)
        os.makedirs(self.full_output_figures_dir)


    def meanAndVariance(self, feature="direct_comparison", hardcopy=True):
        if self.f is None:
            print "Datafile must be loaded"
            sys.exit(1)

        color1 = 0
        color2 = 8

        t = self.f["direct_comparison"]["t"][:]
        E = self.f[feature]["E"][:]
        Var = self.f[feature]["Var"][:]


        prettyPlot(t, E, "Mean, " + self.filename, "time", "voltage", color1)
        plt.savefig(os.path.join(self.full_output_figures_dir, self.filename + "_mean" + self.figureformat))

        prettyPlot(t, Var, "Variance, " + self.filename, "time", "voltage", color2)
        plt.savefig(os.path.join(self.full_output_figures_dir, self.filename + "_variance" + self.figureformat))

        ax, tableau20 = prettyPlot(t, E, "Mean and variance, " + self.filename,
                                   "time", "voltage, mean", color1)
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=tableau20[color2], labelcolor=tableau20[color2], labelsize=14)
        ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
        ax.spines["right"].set_edgecolor(tableau20[color2])

        ax2.set_xlim([min(t), max(t)])
        ax2.set_ylim([min(Var), max(Var)])

        ax2.plot(t, Var, color=tableau20[color2], linewidth=2, antialiased=True)

        ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
        ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
        ax.spines["left"].set_edgecolor(tableau20[color1])
        #plt.tight_layout()

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_figures_dir,
                        feature + "_variance-mean" + self.figureformat))
        else:
            plt.show()

        plt.close()


    def confidenceInterval(self, feature="direct_comparison", hardcopy=True):
        if self.f is None:
            print "Datafile must be loaded"
            sys.exit(1)


        t = self.f["direct_comparison"]["t"][:]
        E = self.f[feature]["E"][:]
        p_05 = self.f[feature]["p_05"][:]
        p_95 = self.f[feature]["p_95"][:]


        ax, color = prettyPlot(t, E, "Confidence interval", "time", "voltage", 0)
        plt.fill_between(t, p_05, p_95, alpha=0.2, facecolor=color[8])
        prettyPlot(t, p_95, color=8, new_figure=False)
        prettyPlot(t, p_05, color=9, new_figure=False)
        prettyPlot(t, E, "Confidence interval", "time", "voltage", 0, False)

        plt.ylim([min([min(p_95), min(p_05), min(E)]),
                  max([max(p_95), max(p_05), max(E)])])

        plt.legend(["Mean", "$P_{95}$", "$P_{5}$"])

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_figures_dir,
                                     feature + "_confidence-interval" + self.figureformat))
        else:
            plt.show()

        plt.close()


    def sensitivity(self, feature="direct_comparison", hardcopy=True):
        if self.f is None:
            print "Datafile must be loaded"
            sys.exit(1)

        if "sensitivity" not in self.f[feature].keys():
            return

        t = self.f["direct_comparison"]["t"][:]
        sensitivity = self.f[feature]["sensitivity"][:]

        parameter_names = self.f.attrs["uncertain parameters"]

        for i in range(len(sensitivity)):
            prettyPlot(t, sensitivity[i],
                       parameter_names[i] + " sensitivity", "time",
                       "sensitivity", i, True)
            plt.title(parameter_names[i] + " sensitivity")
            plt.ylim([0, 1.05])
            plt.savefig(os.path.join(self.full_output_figures_dir,
                                     parameter_names[i] +
                                     "_sensitivity" + self.figureformat))
        plt.close()

        for i in range(len(sensitivity)):
            prettyPlot(t, sensitivity[i], "sensitivity", "time",
                       "sensitivity", i, False)

        plt.ylim([0, 1.05])
        plt.xlim([t[0], 1.3*t[-1]])
        plt.legend(parameter_names)

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_figures_dir,
                                     "all_sensitivity" + self.figureformat))
        else:
            plt.show()

        plt.close()


    def plot2dFeatures(self):
        self.meanAndVariance()
        self.confidenceInterval()
        self.sensitivity()



    def plot1dFeaturesCombined(self, index=0, max_legend_size=5):
        if len(self.f.attrs["uncertain parameters"]) > 8:
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
        figsize = (10, 7.5)


        if len(self.features_1d) > max_legend_size:
            legend_size = max_legend_size
        else:
            legend_size = len(self.f.attrs["uncertain parameters"])

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


            E = self.f[feature_name]["E"][()]
            Var = self.f[feature_name]["Var"][()]
            p_05 = self.f[feature_name]["p_05"][()]
            p_95 = self.f[feature_name]["p_95"][()]

            if "sensitivity" in self.f[feature_name].keys():
                sensitivity = self.f[feature_name]["sensitivity"][:]

            ax.bar(pos, E, yerr=Var, width=width, align='center', color=tableau20[0], linewidth=0,
                   error_kw=dict(ecolor=axis_grey, lw=2, capsize=5, capthick=2))
            xticks.append(pos)
            xticklabels.append("mean")


            pos += distance

            ax.bar(pos, p_05, width=width, align='center', color=tableau20[3], linewidth=0)
            ax.bar(pos + width, p_95, width=width, align='center', color=tableau20[2], linewidth=0)
            xticks += [pos, pos + width]
            xticklabels += ["$P_5$", "$P_{95}$"]

            if "sensitivity" in self.f[feature_name].keys():
                pos += distance + width

                i = 0
                legend_bars = []

                for parameter in self.f.attrs["uncertain parameters"]:
                    # TODO is abs(sensitivity) a problem in the plot?
                    legend_bars.append(ax2.bar(pos, abs(sensitivity[i]), width=width, align='center', color=tableau20[4+i], linewidth=0))

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
            else:
                # TODO is abs(sensitivity) a problem in the plot?
                # ax2.bar(pos, abs(sensitivity), width=width, align='center', color=tableau20[4], linewidth=0)
                xticks.append(pos + distance)
                xticklabels.append("")

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45)
            ax.set_title(feature_name)

            ax_i += 1


        ax_all[0].set_ylabel('Feature value', fontsize=fontsize)
        ax2.set_ylabel('Sensitivity', fontsize=fontsize, color=tableau20[4])

        if "sensitivity" in self.f[feature_name].keys():
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


            lgd = ax.legend(legend_bars, self.f.attrs["uncertain parameters"], loc=loc, bbox_to_anchor=location,
                            fancybox=False, shadow=False, ncol=legend_size)
            lgd.get_frame().set_edgecolor(axis_grey)

            # plt.tight_layout()
            fig.subplots_adjust(top=(0.91 - legend_width*0.053), wspace=0.5)
        else:
            fig.subplots_adjust(wspace=0.5)

        plt.suptitle(self.filename, fontsize=titlesize)

        save_name = self.filename + ("_features_%d" % (index/self.features_in_combined_plot)) + self.figureformat
        plt.savefig(os.path.join(self.full_output_figures_dir, save_name))
        plt.close()


    def plot1dFeatures(self):
        for parameter_name in self.f.keys():
            for feature_name in self.f.attrs["features"]:
                self.plotFeature(feature_name=feature_name, parameter_name=parameter_name)
                save_name = "%s_%s" % (parameter_name, feature_name) + self.figureformat
                plt.savefig(os.path.join(self.full_output_figures_dir, save_name))
                plt.close()


    # TODO not finhised, missing correct label placement
    def plot1dFeature(self, feature_name, max_legend_size=5):
        if feature_name not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature_name))
        E = self.f[feature_name]["E"][()]
        Var = self.f[feature_name]["Var"][()]
        p_05 = self.f[feature_name]["p_05"][()]
        p_95 = self.f[feature_name]["p_95"][()]

        if parameter_name == "all":
            sensitivity = self.f[parameter_name][feature_name]["sensitivity"][:]
        else:
            sensitivity = self.f[parameter_name][feature_name]["sensitivity"][()]


        if len(self.f.attrs["uncertain parameters"]) > max_legend_size:
            legend_size = max_legend_size
        else:
            legend_size = len(self.f.attrs["uncertain parameters"])

        legend_width = np.ceil(len(self.f.attrs["uncertain parameters"])/float(max_legend_size))

        axis_grey = (0.5, 0.5, 0.5)
        titlesize = 18
        fontsize = 16
        labelsize = 14
        figsize = (10, 7.5)

        width = 0.2
        distance = 0.5

        pos = 0
        xticks = [pos]
        xticklabels = ["mean"]
        ax, tableau20 = prettyBar(E, Var)
        ax.spines["right"].set_edgecolor(axis_grey)

        pos += distance

        ax.set_ylabel("Feature value", fontsize=labelsize)

        ax.bar(pos, p_05, width=width, align='center', color=tableau20[3], linewidth=0)
        ax.bar(pos + width, p_95, width=width, align='center', color=tableau20[2], linewidth=0)
        xticks += [pos, pos + width]
        xticklabels += ["$P_5$", "$P_{95}$"]
        pos += distance + width

        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=axis_grey, labelcolor=tableau20[4], labelsize=labelsize)
        ax2.set_ylabel('Sensitivity', fontsize=fontsize, color=tableau20[4])
        ax2.set_ylim([0, 1.05])


        if parameter_name == "all":
            i = 0
            legend_bars = []
            for parameter in self.f.attrs["uncertain parameters"]:
                legend_bars.append(ax2.bar(pos, sensitivity[i], width=width,
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
            lgd = plt.legend(legend_bars, self.f.attrs["uncertain parameters"],
                             loc='upper center', bbox_to_anchor=location,
                             fancybox=False, shadow=False, ncol=legend_size)
            lgd.get_frame().set_edgecolor(axis_grey)

            fig = plt.gcf()
            fig.subplots_adjust(top=(0.91 - legend_width*0.053))

        else:
            ax.bar(pos, sensitivity, width=width, align='center', color=tableau20[4], linewidth=0)
            xticks.append(pos)
            xticklabels.append(parameter_name)


        pos += 3*distance
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45)

        plt.suptitle("Parameter: " + parameter_name + ", feature: " + feature_name, fontsize=titlesize)

        return ax, tableau20, pos



    def plotAllData(self, combined_features=False):
        print "Plotting all data"

        for f in glob.glob(os.path.join(self.data_dir, "*")):
            self.loadData(f.split("/")[-1])

            self.plot2dFeatures()

            if combined_features:
                self.plot1dFeaturesCombined()
            else:
                self.plotFeatures()



    # def gif(self):
    #     print "Creating gifs..."
    #
    #     if os.path.isdir(self.output_gif_dir):
    #         shutil.rmtree(self.output_gif_dir)
    #     os.makedirs(self.output_gif_dir)
    #
    #     plot_types = ["mean", "variance", "variance-mean", "confidence-interval", "sensitivity"]
    #
    #     files = {}
    #     plotting_order = {}
    #     for f in glob.glob(self.data_dir + "*"):
    #         if re.search("._\d", f):
    #             files[f.split("/")[-1]] = h5py.File(f, 'r')
    #
    #             distribution, interval = f.split("/")[-1].split("_")
    #
    #             if distribution in plotting_order:
    #                 plotting_order[distribution].append(interval)
    #             else:
    #                 plotting_order[distribution] = [interval]
    #
    #
    #     uncertain_parameters = []
    #     for parameter in files[distribution + "_" + interval].keys():
    #         uncertain_parameters.append(parameter)
    #
    #
    #     # Run through each distribution and plot everything for each distribution
    #     for distribution in plotting_order:
    #         if os.path.isdir(self.tmp_gif_output):
    #             shutil.rmtree(self.tmp_gif_output)
    #         os.makedirs(self.tmp_gif_output)
    #
    #         # Create the plot for each parameter
    #         for parameter in uncertain_parameters:
    #
    #             # Finding the max and min data point for all distributions for each parameter
    #             max_data = {}
    #             min_data = {}
    #             for interval in plotting_order[distribution]:
    #                 filename = distribution + "_" + interval
    #                 f = files[filename]
    #
    #                 for datasett in f[parameter].keys():
    #                     if datasett == "sensitivity" or datasett == "total sensitivity":
    #                         continue
    #
    #                     max_value = max(f[parameter][datasett])
    #                     min_value = min(f[parameter][datasett])
    #
    #
    #                     if datasett in max_data:
    #                         if max_value > max_data[datasett]:
    #                             max_data[datasett] = max_value
    #                         if min_value > min_data[datasett]:
    #                             min_data[datasett] = min_value
    #                     else:
    #                         max_data[datasett] = max_value
    #                         min_data[datasett] = min_value
    #
    #
    #             for interval in plotting_order[distribution]:
    #                 filename = distribution + "_" + interval
    #                 f = files[filename]
    #
    #                 t = f[parameter]["t"][:]
    #                 E = f[parameter]["E"][:]
    #                 Var = f[parameter]["Var"][:]
    #                 p_05 = f[parameter]["p_05"][:]
    #                 p_95 = f[parameter]["p_95"][:]
    #
    #
    #                 # Create the different plots
    #
    #
    #                 ### V(t)
    #                 color1 = 0
    #                 color2 = 8
    #
    #                 title = parameter + ": Mean, " + distribution + " " + interval
    #                 #plt.rcParams["figure.figsize"] = (10, 7.5)
    #                 prettyPlot(t, E, title, "time", "voltage", color1)
    #                 plt.ylim(min_data["E"], max_data["E"])
    #                 plt.xlim(min_data["t"], max_data["t"])
    #                 save_name = parameter + "_" + interval + "_mean" + self.figureformat
    #                 plt.savefig(os.path.join(self.tmp_gif_output, save_name))
    #                 plt.close()
    #
    #                 title = parameter + ": Variance, " + distribution + " " + interval
    #                 prettyPlot(t, Var, title, "time", "voltage", color2)
    #                 plt.ylim(min_data["Var"], max_data["Var"])
    #                 plt.xlim(min_data["t"], max_data["t"])
    #                 save_name = parameter + "_" + interval + "_variance" + self.figureformat
    #                 plt.savefig(os.path.join(self.tmp_gif_output, save_name))
    #                 plt.close()
    #
    #                 title = parameter + ": Mean and variance, " + distribution + " " + interval
    #                 ax, tableau20 = prettyPlot(t, E, title, "time", "voltage, mean", color1)
    #                 plt.ylim([min_data["E"], max_data["E"]])
    #                 ax2 = ax.twinx()
    #                 ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
    #                                 color=tableau20[color2], labelcolor=tableau20[color2], labelsize=14)
    #                 ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
    #                 ax.spines["right"].set_edgecolor(tableau20[color2])
    #
    #                 ax2.set_xlim([min_data["t"], max_data["t"]])
    #                 ax2.set_ylim([min_data["Var"], max_data["Var"]])
    #
    #                 ax2.plot(t, Var, color=tableau20[color2], linewidth=2, antialiased=True)
    #
    #                 ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
    #                 ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
    #                 ax.spines["left"].set_edgecolor(tableau20[color1])
    #
    #
    #                 save_name = parameter + "_" + interval + "_variance-mean" + self.figureformat
    #                 plt.savefig(os.path.join(self.tmp_gif_output, save_name))
    #                 plt.close()
    #
    #
    #                 ### Confidence interval
    #                 title = parameter + ": Confidence interval, " + distribution + " " + interval
    #                 ax, color = prettyPlot(t, E, title, "voltage", 0)
    #                 plt.fill_between(t, p_05, p_95, alpha=0.2, facecolor=color[8])
    #                 prettyPlot(t, p_95, color=8, new_figure=False)
    #                 prettyPlot(t, p_05, color=9, new_figure=False)
    #                 prettyPlot(t, E, title, "time", "voltage", 0, False)
    #
    #                 tmp_min = min([min_data["p_95"], min_data["p_05"], min_data["E"]])
    #                 tmp_max = max([max_data["p_95"], max_data["p_05"], max_data["E"]])
    #                 plt.ylim(tmp_min, tmp_max)
    #
    #                 plt.legend(["Mean", "$P_{95}$", "$P_{5}$"])
    #
    #                 save_name = parameter + "_" + interval + "_confidence-interval" + self.figureformat
    #                 plt.savefig(os.path.join(self.tmp_gif_output, save_name))
    #
    #                 if parameter == "all":
    #                     sensitivity = f[parameter]["sensitivity"][:]
    #
    #                     sensitivity_parameters = f.attrs["uncertain parameters"]
    #                     for i in range(len(sensitivity_parameters)):
    #                         title = sensitivity_parameters[i] + ": Sensitivity, " + distribution + " " + interval
    #                         prettyPlot(t, sensitivity[i], title, "time", "sensitivity", i, True)
    #                         plt.ylim([0, 1.05])
    #                         save_name = sensitivity_parameters[i] + "_" + interval + "_sensitivity" + self.figureformat
    #                         plt.savefig(os.path.join(self.tmp_gif_output, save_name))
    #                         plt.close()
    #
    #                     for i in range(len(sensitivity_parameters)):
    #                         title = "all: Sensitivity, " + distribution + " " + interval
    #                         prettyPlot(t, sensitivity[i], title, "time",
    #                                    "sensitivity", i, False)
    #
    #                     plt.ylim([0, 1.05])
    #                     plt.xlim([t[0], 1.3*t[-1]])
    #                     plt.legend(sensitivity_parameters)
    #                     save_name = "all_" + interval + "_sensitivity" + self.figureformat
    #                     plt.savefig(os.path.join(self.tmp_gif_output, save_name))
    #                     plt.close()
    #
    #
    #         # Create gif
    #         outputdir = os.path.join(self.output_gif_dir, distribution)
    #         if os.path.isdir(outputdir):
    #             shutil.rmtree(outputdir)
    #         os.makedirs(outputdir)
    #
    #         for parameter in uncertain_parameters:
    #             for plot_type in plot_types:
    #                 final_name = os.path.join(outputdir, parameter + "_" + plot_type)
    #                 file_name = os.path.join(self.tmp_gif_output, "%s_*_%s%s" % (parameter, plot_type, self.figureformat))
    #                 cmd = "convert -set delay 100 %s %s.gif" % (file_name, final_name)
    #
    #                 os.system(cmd)
    #
    #         shutil.rmtree(self.tmp_gif_output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot data")
    parser.add_argument("-d", "--data_dir", help="Directory the data is stored in")

    args = parser.parse_args()

    data_dir = "data/"
    output_figures_dir = "figures/"
    output_gif_dir = "gifs/"
    figureformat = ".png"

    if args.data_dir:
        data_dir = "%s/" % os.path.join(data_dir, args.data_dir)
        output_figures_dir = "%s/" % os.path.join(output_figures_dir, args.data_dir)
        output_gif_dir = "%s/" % os.path.join(output_gif_dir, args.data_dir)


    plot = PlotUncertainty(data_dir=data_dir,
                           output_figures_dir=output_figures_dir,
                           figureformat=figureformat,
                           output_gif_dir=output_gif_dir)

    plot.plotAllData()
    #plot.gif()

    # sortByParameters(path=output_figures_dir, outputpath=output_figures_dir)
