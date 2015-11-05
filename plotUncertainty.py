import os
import h5py
import sys
import shutil
import glob

import matplotlib.pyplot as plt

from collect_by_parameter import saveByParameters, createGIF
from prettyPlot import prettyPlot



class PlotUncertainty():
    def __init__(self,
                 data_dir="data/",
                 output_figures_dir="figures/",
                 figureformat=".png"):

        self.data_dir = data_dir
        self.output_figures_dir = output_figures_dir
        self.figureformat = figureformat
        self.f = None


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
        plt.savefig(os.path.join(self.full_output_figures_dir, parameter + "_mean" + self.figureformat),
                    bbox_inches="tight")

        prettyPlot(t, Var, "Variance, " + parameter, "time", "voltage", color2)
        plt.savefig(os.path.join(self.full_output_figures_dir, parameter + "_variance" + self.figureformat),
                    bbox_inches="tight")

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
        plt.tight_layout()

        plt.savefig(os.path.join(self.full_output_figures_dir,
                    parameter + "_variance-mean" + self.figureformat),
                    bbox_inches="tight")

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
        plt.savefig(os.path.join(self.full_output_figures_dir, parameter + "_confidence-interval" + self.figureformat),
                    bbox_inches="tight")

        plt.close()

    def sensitivity(self):

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
                                     "_sensitivity" + self.figureformat),
                        bbox_inches="tight")
        plt.close()

        for i in range(len(sensitivity)):
            prettyPlot(t, sensitivity[i], "sensitivity", "time",
                       "sensitivity", i, False)

        plt.ylim([0, 1.05])
        plt.xlim([t[0], 1.3*t[-1]])
        plt.legend(parameter_names)
        plt.savefig(os.path.join(self.full_output_figures_dir,
                                 "all_sensitivity" + self.figureformat),
                    bbox_inches="tight")


    def all(self):
        for parameter in self.f.keys():
            self.vt(parameter)
            self.confidenceInterval(parameter)

        self.sensitivity()

    def allData(self):
        for f in glob.glob(self.data_dir + "*"):
            self.loadData(f.split("/")[-1])
            self.all()

if __name__ == "__main__":
    data_dir = "data/"
    output_figures_dir = "figures/"
    figureformat = ".png"

    plot = PlotUncertainty(data_dir=data_dir,
                           output_figures_dir=output_figures_dir,
                           figureformat=figureformat)

    #plot.loadData("uniform_0.05")
    plot.allData()
