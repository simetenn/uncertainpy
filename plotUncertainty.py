import os
import h5py
import sys
import shutil
import glob
import re

import matplotlib.pyplot as plt

from prettyPlot import prettyPlot
from collect_by_parameter import sortByParameters

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



    def vt(self, parameter):
        if self.f is None:
            print "Datafile must be loaded"
            sys.exit(1)

        color1 = 0
        color2 = 8


        t = self.f[parameter]["t"][:]
        E = self.f[parameter]["E"][:]
        Var = self.f[parameter]["Var"][:]


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


    def confidenceInterval(self, parameter):

        t = self.f[parameter]["t"][:]
        E = self.f[parameter]["E"][:]
        p_05 = self.f[parameter]["p_05"][:]
        p_95 = self.f[parameter]["p_95"][:]


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

    def sensitivity(self, hardcopy=True):

        t = self.f["all"]["t"][:]
        sensitivity = self.f["all"]["sensitivity"][:]

        parameter_names = self.f.attrs["Uncertain parameters"]

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

    def all(self):
        for parameter in self.f.keys():
            self.vt(parameter)
            self.confidenceInterval(parameter)

        self.sensitivity()

    def allData(self):
        print "Plotting all data"
        for f in glob.glob(self.data_dir + "*"):
            self.loadData(f.split("/")[-1])
            self.all()


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

                        sensitivity_parameters = f.attrs["Uncertain parameters"]
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
    data_dir = "data/"
    output_figures_dir = "figures/"
    figureformat = ".png"
    output_gif_dir = "gifs/"

    plot = PlotUncertainty(data_dir=data_dir,
                           output_figures_dir=output_figures_dir,
                           figureformat=figureformat,
                           output_gif_dir=output_gif_dir)

    plot.allData()
    plot.gif()
    sortByParameters()
