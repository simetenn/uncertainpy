import os
import h5py
import sys
import shutil
import glob
import re
import argparse

import matplotlib.pyplot as plt
import numpy as np

from prettyPlot import prettyPlot, prettyBar
from uncertainpy.utils import sortByParameters

### TODO rewrite gif() to use less memory when creating GIF(Only load one dataset at the time)

### TODO refactor gif() to be less complex and use more functions

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

    def loadData(self, filename):
        self.f = h5py.File(os.path.join(self.data_dir, filename), 'r')

        self.full_output_figures_dir = os.path.join(self.output_figures_dir, filename)

        if os.path.isdir(self.full_output_figures_dir):
            shutil.rmtree(self.full_output_figures_dir)
        os.makedirs(self.full_output_figures_dir)



    def vt(self, parameter="all", feature="direct_comparison"):
        if self.f is None:
            print "Datafile must be loaded"
            sys.exit(1)

        color1 = 0
        color2 = 8


        t = self.f[parameter][feature]["t"][:]
        E = self.f[parameter][feature]["E"][:]
        Var = self.f[parameter][feature]["Var"][:]


        prettyPlot(t, E, "Mean, " + parameter, "time", "voltage", color1)
        plt.savefig(os.path.join(self.full_output_figures_dir, parameter + "_mean" + self.figureformat))

        prettyPlot(t, Var, "Variance, " + parameter, "time", "voltage", color2)
        plt.savefig(os.path.join(self.full_output_figures_dir, parameter + "_variance" + self.figureformat))

        ax, tableau20 = prettyPlot(t, E, "Mean and variance, " + parameter,
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

        plt.savefig(os.path.join(self.full_output_figures_dir,
                    parameter + "_variance-mean" + self.figureformat))

        plt.close()


    def confidenceInterval(self, parameter="all", feature="direct_comparison"):
        if self.f is None:
            print "Datafile must be loaded"
            sys.exit(1)


        t = self.f[parameter][feature]["t"][:]
        E = self.f[parameter][feature]["E"][:]
        p_05 = self.f[parameter][feature]["p_05"][:]
        p_95 = self.f[parameter][feature]["p_95"][:]


        ax, color = prettyPlot(t, E, "Confidence interval", "time", "voltage", 0)
        plt.fill_between(t, p_05, p_95, alpha=0.2, facecolor=color[8])
        prettyPlot(t, p_95, color=8, new_figure=False)
        prettyPlot(t, p_05, color=9, new_figure=False)
        prettyPlot(t, E, "Confidence interval", "time", "voltage", 0, False)

        plt.ylim([min([min(p_95), min(p_05), min(E)]),
                  max([max(p_95), max(p_05), max(E)])])

        plt.legend(["Mean", "$P_{95}$", "$P_{5}$"])

        plt.savefig(os.path.join(self.full_output_figures_dir, parameter + "_confidence-interval" + self.figureformat))
        plt.close()

    def sensitivity(self, feature="direct_comparison", hardcopy=True):
        if self.f is None:
            print "Datafile must be loaded"
            sys.exit(1)

        t = self.f["all"][feature]["t"][:]
        sensitivity = self.f["all"][feature]["sensitivity"][:]

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
        plt.savefig(os.path.join(self.full_output_figures_dir,
                                 "all_sensitivity" + self.figureformat))
        plt.close()

    def plotAllDirectComparison(self):
        for parameter in self.f.keys():
            self.vt(parameter)
            self.confidenceInterval(parameter)

        self.sensitivity()



    # TODO fix title, and names for each feature
    def plotFeatures(self):
        feature_names = self.f.attrs["features"]
        #parameter_names = self.f.attrs["uncertain parameters"]

        axis_grey = (0.5, 0.5, 0.5)
        titlesize = 18
        fontsize = 16
        labelsize = 14
        figsize = (10, 7.5)

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

        plt.rcParams["figure.figsize"] = figsize


        for parameter_name in self.f.keys():
            pos = 0
            xticks = []
            xticklabels = []

            plt.figure(figsize=figsize)
            ax = plt.subplot(111)

            ax.spines["top"].set_edgecolor("None")
            ax.spines["bottom"].set_edgecolor(axis_grey)
            ax.spines["right"].set_edgecolor("None")
            ax.spines["left"].set_edgecolor(axis_grey)


            ax.tick_params(axis="x", which="both", bottom="on", top="off",
                           labelbottom="on", color=axis_grey, labelcolor="black",
                           labelsize=labelsize)
            ax.tick_params(axis="y", which="both", right="off", left="on",
                           labelleft="on", color=axis_grey, labelcolor="black",
                           labelsize=labelsize)

            a = []
            b = []

            for feature_name in feature_names:
                E = self.f[parameter_name][feature_name]["E"][()]
                Var = self.f[parameter_name][feature_name]["Var"][()]
                p_05 = self.f[parameter_name][feature_name]["p_05"][()]
                p_95 = self.f[parameter_name][feature_name]["p_95"][()]

                if parameter_name == "all":
                    sensitivity = self.f[parameter_name][feature_name]["sensitivity"][:]
                else:
                    sensitivity = self.f[parameter_name][feature_name]["sensitivity"][()]

                ax.bar(pos, E, yerr=Var, width=width, align='center', color=tableau20[0], linewidth=0,
                       error_kw=dict(ecolor=axis_grey, lw=2, capsize=5, capthick=2))
                xticks.append(pos)
                xticklabels.append("mean")


                pos += distance

                ax.bar(pos, p_05, width=width, align='center', color=tableau20[3], linewidth=0)
                ax.bar(pos + width, p_95, width=width, align='center', color=tableau20[2], linewidth=0)
                xticks += [pos, pos + width]
                xticklabels += ["$P_5$", "$P_{95}$"]
                pos += distance + width

                if parameter_name == "all":
                    i = 0
                    ax.text(pos, -5, "Sensitivity", fontsize=labelsize)
                    a.append(pos)
                    b.append("sensitivity")
                    for parameter in self.f.attrs["uncertain parameters"]:
                        # TODO is abs(sensitivity) a problem in the plot?
                        ax.bar(pos, abs(sensitivity[i]), width=width, align='center', color=tableau20[4+i], linewidth=0)
                        xticks.append(pos)
                        xticklabels.append(parameter)

                        i += 1
                        pos += width

                else:
                    # TODO is abs(sensitivity) a problem in the plot?
                    ax.bar(pos, abs(sensitivity), width=width, align='center', color=tableau20[4], linewidth=0)
                    xticks.append(pos)
                    xticklabels.append("Sensitivity")

                pos += 2*distance

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45)
            # ax.get_xaxis().set_tick_params(which='Minor', pad=10)
            # ax.set_xticks(a, minor=True)
            # ax.set_xticklabels(b, minor=True, fontsize=labelsize)

            #ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45)

            ax.set_title("Features for %s" % parameter_name)
            ax.set_legend()
            plt.show()
            plt.close()


    def plotFeature(self, feature_name, parameter_name="all"):
        E = self.f[parameter_name][feature_name]["E"][()]
        Var = self.f[parameter_name][feature_name]["Var"][()]
        p_05 = self.f[parameter_name][feature_name]["p_05"][()]
        p_95 = self.f[parameter_name][feature_name]["p_95"][()]

        if parameter_name == "all":
            sensitivity = self.f[parameter_name][feature_name]["sensitivity"][:]
        else:
            sensitivity = self.f[parameter_name][feature_name]["sensitivity"][()]

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

        pos += distance

        ax.bar(pos, p_05, width=width, align='center', color=tableau20[3], linewidth=0)
        ax.bar(pos + width, p_95, width=width, align='center', color=tableau20[2], linewidth=0)
        xticks += [pos, pos + width]
        xticklabels += ["$P_5$", "$P_{95}$"]
        pos += distance + width

        if parameter_name == "all":
            i = 0
            for parameter in self.f.attrs["uncertain parameters"]:
                ax.bar(pos, sensitivity[i], width=width, align='center', color=tableau20[4+i], linewidth=0)
                xticks.append(pos)
                xticklabels.append(parameter)

                i += 1
                pos += width
        else:
            print sensitivity
            ax.bar(pos, sensitivity, width=width, align='center', color=tableau20[4], linewidth=0)
            xticks.append(pos)
            xticklabels.append(parameter_name)

        pos += 3*distance

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=labelsize, rotation=-45)

        return ax, tableau20, pos



    def plotAllData(self):
        print "Plotting all data"
        for f in glob.glob(self.data_dir + "*"):
            self.loadData(f.split("/")[-1])
            self.plotAllDirectComparison()
            self.plotFeatures()


    def gif(self):
        print "Creating gifs..."

        if os.path.isdir(self.output_gif_dir):
            shutil.rmtree(self.output_gif_dir)
        os.makedirs(self.output_gif_dir)

        plot_types = ["mean", "variance", "variance-mean", "confidence-interval", "sensitivity"]

        files = {}
        plotting_order = {}
        for f in glob.glob(self.data_dir + "*"):
            if re.search("._\d", f):
                files[f.split("/")[-1]] = h5py.File(f, 'r')

                distribution, interval = f.split("/")[-1].split("_")

                if distribution in plotting_order:
                    plotting_order[distribution].append(interval)
                else:
                    plotting_order[distribution] = [interval]


        uncertain_parameters = []
        for parameter in files[distribution + "_" + interval].keys():
            uncertain_parameters.append(parameter)


        # Run through each distribution and plot everything for each distribution
        for distribution in plotting_order:
            if os.path.isdir(self.tmp_gif_output):
                shutil.rmtree(self.tmp_gif_output)
            os.makedirs(self.tmp_gif_output)

            # Create the plot for each parameter
            for parameter in uncertain_parameters:

                # Finding the max and min data point for all distributions for each parameter
                max_data = {}
                min_data = {}
                for interval in plotting_order[distribution]:
                    filename = distribution + "_" + interval
                    f = files[filename]

                    for datasett in f[parameter].keys():
                        if datasett == "sensitivity" or datasett == "total sensitivity":
                            continue

                        max_value = max(f[parameter][datasett])
                        min_value = min(f[parameter][datasett])


                        if datasett in max_data:
                            if max_value > max_data[datasett]:
                                max_data[datasett] = max_value
                            if min_value > min_data[datasett]:
                                min_data[datasett] = min_value
                        else:
                            max_data[datasett] = max_value
                            min_data[datasett] = min_value


                for interval in plotting_order[distribution]:
                    filename = distribution + "_" + interval
                    f = files[filename]

                    t = f[parameter]["t"][:]
                    E = f[parameter]["E"][:]
                    Var = f[parameter]["Var"][:]
                    p_05 = f[parameter]["p_05"][:]
                    p_95 = f[parameter]["p_95"][:]


                    # Create the different plots


                    ### V(t)
                    color1 = 0
                    color2 = 8

                    title = parameter + ": Mean, " + distribution + " " + interval
                    #plt.rcParams["figure.figsize"] = (10, 7.5)
                    prettyPlot(t, E, title, "time", "voltage", color1)
                    plt.ylim(min_data["E"], max_data["E"])
                    plt.xlim(min_data["t"], max_data["t"])
                    save_name = parameter + "_" + interval + "_mean" + self.figureformat
                    plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                    plt.close()

                    title = parameter + ": Variance, " + distribution + " " + interval
                    prettyPlot(t, Var, title, "time", "voltage", color2)
                    plt.ylim(min_data["Var"], max_data["Var"])
                    plt.xlim(min_data["t"], max_data["t"])
                    save_name = parameter + "_" + interval + "_variance" + self.figureformat
                    plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                    plt.close()

                    title = parameter + ": Mean and variance, " + distribution + " " + interval
                    ax, tableau20 = prettyPlot(t, E, title, "time", "voltage, mean", color1)
                    plt.ylim([min_data["E"], max_data["E"]])
                    ax2 = ax.twinx()
                    ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                                    color=tableau20[color2], labelcolor=tableau20[color2], labelsize=14)
                    ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
                    ax.spines["right"].set_edgecolor(tableau20[color2])

                    ax2.set_xlim([min_data["t"], max_data["t"]])
                    ax2.set_ylim([min_data["Var"], max_data["Var"]])

                    ax2.plot(t, Var, color=tableau20[color2], linewidth=2, antialiased=True)

                    ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
                    ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
                    ax.spines["left"].set_edgecolor(tableau20[color1])


                    save_name = parameter + "_" + interval + "_variance-mean" + self.figureformat
                    plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                    plt.close()


                    ### Confidence interval
                    title = parameter + ": Confidence interval, " + distribution + " " + interval
                    ax, color = prettyPlot(t, E, title, "voltage", 0)
                    plt.fill_between(t, p_05, p_95, alpha=0.2, facecolor=color[8])
                    prettyPlot(t, p_95, color=8, new_figure=False)
                    prettyPlot(t, p_05, color=9, new_figure=False)
                    prettyPlot(t, E, title, "time", "voltage", 0, False)

                    tmp_min = min([min_data["p_95"], min_data["p_05"], min_data["E"]])
                    tmp_max = max([max_data["p_95"], max_data["p_05"], max_data["E"]])
                    plt.ylim(tmp_min, tmp_max)

                    plt.legend(["Mean", "$P_{95}$", "$P_{5}$"])

                    save_name = parameter + "_" + interval + "_confidence-interval" + self.figureformat
                    plt.savefig(os.path.join(self.tmp_gif_output, save_name))

                    if parameter == "all":
                        sensitivity = f[parameter]["sensitivity"][:]

                        sensitivity_parameters = f.attrs["uncertain parameters"]
                        for i in range(len(sensitivity_parameters)):
                            title = sensitivity_parameters[i] + ": Sensitivity, " + distribution + " " + interval
                            prettyPlot(t, sensitivity[i], title, "time", "sensitivity", i, True)
                            plt.ylim([0, 1.05])
                            save_name = sensitivity_parameters[i] + "_" + interval + "_sensitivity" + self.figureformat
                            plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                            plt.close()

                        for i in range(len(sensitivity_parameters)):
                            title = "all: Sensitivity, " + distribution + " " + interval
                            prettyPlot(t, sensitivity[i], title, "time",
                                       "sensitivity", i, False)

                        plt.ylim([0, 1.05])
                        plt.xlim([t[0], 1.3*t[-1]])
                        plt.legend(sensitivity_parameters)
                        save_name = "all_" + interval + "_sensitivity" + self.figureformat
                        plt.savefig(os.path.join(self.tmp_gif_output, save_name))
                        plt.close()


            # Create gif
            outputdir = os.path.join(self.output_gif_dir, distribution)
            if os.path.isdir(outputdir):
                shutil.rmtree(outputdir)
            os.makedirs(outputdir)

            for parameter in uncertain_parameters:
                for plot_type in plot_types:
                    final_name = os.path.join(outputdir, parameter + "_" + plot_type)
                    file_name = os.path.join(self.tmp_gif_output, "%s_*_%s%s" % (parameter, plot_type, self.figureformat))
                    cmd = "convert -set delay 100 %s %s.gif" % (file_name, final_name)

                    os.system(cmd)

            shutil.rmtree(self.tmp_gif_output)


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

    sortByParameters(path=output_figures_dir, outputpath=output_figures_dir)
