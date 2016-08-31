import os
import h5py
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Fix Remove import * once finished
from uncertainpy.plotting.prettyPlot import prettyPlot, prettyBar
from uncertainpy.plotting.prettyPlot import spines_edge_color, get_current_colormap
from uncertainpy.plotting.prettyPlot import set_legend, get_colormap_tableu20
from uncertainpy.plotting.prettyPlot import axis_grey, labelsize, fontsize, titlesize
from uncertainpy.utils import create_logger

# TODO rewrite gif() to use less memory when creating GIF
# (Only load one dataset at the time)

# TODO Add feature plots to gif()

# TODO find a good way to find the directory where the data files are


# TODO move load() to it's own class

# TODO compare plots in a grid of all plots,
# such as plotting all features in a grid plot

# TODO Move some parts of the ploting code into it's own class for increased
# readability

# TODO CHange the use of **Kwargs to use a dict for specific plotting commands?


# TODO plot simulator_results

class PlotUncertainty():
    def __init__(self,
                 data_dir="data/",
                 output_dir_figures="figures/",
                 figureformat=".png",
                 verbose_level="info",
                 verbose_filename=None,
                 xlabel="time [ms]",
                 ylabel="voltage [mv]"):

        self.data_dir = data_dir
        self.output_dir_figures = output_dir_figures
        self.figureformat = figureformat
        self.f = None

        self.tmp_gif_output = ".tmp_gif_output/"
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.features_in_combined_plot = 3

        self.loaded_flag = False

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        self.features_0d = []
        self.features_1d = []

        self.xlabel = xlabel
        self.ylabel = ylabel


    def loadData(self, filename, create_output_folder=True):
        self.filename = filename
        full_path = os.path.join(self.data_dir, self.filename)

        f = h5py.File(full_path, 'r')

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

        if not os.path.isdir(self.full_output_dir_figures) and create_output_folder:
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

    def toLatex(self, text):
        if "_" in text:
            txt = text.split("_")
            return "$" + txt[0] + "_{" + "-".join(txt[1:]) + "}$"
        else:
            return text

    def listToLatex(self, texts):
        tmp = []
        for txt in texts:
            tmp.append(self.toLatex(txt))

        return tmp

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


    def plotAttributeFeature1d(self, feature="directComparison",
                               attribute="E", attribute_name="mean",
                               hardcopy=True, show=False,
                               **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))



        value = getattr(self, attribute)
        title = feature + ", " + attribute_name
        prettyPlot(self.t[feature], value[feature],
                   title, self.xlabel, self.ylabel, **kwargs)


        save_name = feature + "_" + attribute_name

        if hardcopy:
            plt.savefig(os.path.join(self.full_output_dir_figures,
                                     save_name + self.figureformat),
                        bbox_inches="tight")
            if not show:
                plt.close()

        if show:
            plt.show()

    def plotMean(self, feature, hardcopy=True,
                 show=False, **kwargs):
        self.plotAttributeFeature1d(feature, attribute="E",
                                    attribute_name="mean",
                                    hardcopy=hardcopy,
                                    show=show, **kwargs)


    def plotVariance(self, feature, hardcopy=True,
                     show=False, **kwargs):
        self.plotAttributeFeature1d(feature, attribute="Var",
                                    attribute_name="variance",
                                    hardcopy=hardcopy,
                                    show=show, **kwargs)



    def plotMeanAndVariance(self, feature="directComparison", new_figure=True,
                            hardcopy=True, show=False, color=0, sns_style="dark",
                            **kwargs):
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        ax = prettyPlot(self.t[feature], self.E[feature],
                        feature + ", mean and variance", self.xlabel, self.ylabel + ", mean",
                        sns_style=sns_style, **kwargs)

        colors = get_current_colormap()

        ax2 = ax.twinx()

        spines_edge_color(ax2, edges={"top": "None", "bottom": "None",
                                      "right": colors[color+1], "left": "None"})
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=colors[color+1], labelcolor=colors[color+1], labelsize=labelsize)
        ax2.set_ylabel(self.ylabel + ', variance', color=colors[color+1], fontsize=labelsize)

        # ax2.set_xlim([min(self.t[feature]), max(self.t[feature])])
        # ax2.set_ylim([min(self.Var[feature]), max(self.Var[feature])])

        ax2.plot(self.t[feature], self.Var[feature],
                 color=colors[color+1], linewidth=2, antialiased=True)

        ax2.yaxis.offsetText.set_fontsize(16)
        ax2.yaxis.offsetText.set_color(colors[color+1])


        ax.tick_params(axis="y", color=colors[color], labelcolor=colors[color])
        ax.spines["left"].set_edgecolor(colors[color])

        ax.set_ylabel(self.ylabel + ', mean', color=colors[color], fontsize=16)


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

        prettyPlot(self.t[feature], self.E[feature], title=feature + ", 90\\% confidence interval",
                   xlabel=self.xlabel, ylabel=self.ylabel, color=0,
                   **kwargs)

        colors = get_current_colormap()
        plt.fill_between(self.t[feature], self.p_05[feature], self.p_95[feature],
                         alpha=0.5, color=colors[0])


        set_legend(["mean", "90\% confidence interval"])



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
                       title=feature + ", sensitivity, " + self.toLatex(parameter_names[i]),
                       xlabel=self.xlabel, ylabel="sensitivity",
                       color=i, new_figure=True,
                       nr_hues=len(self.uncertain_parameters), **kwargs)
            # plt.ylim([0, 1.05])

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
                       title=self.toLatex(parameter_names[i]), color=i,
                       nr_hues=nr_plots, ax=ax,
                       **kwargs)
            ax.set_ylabel("sensitivity", fontsize=10)
            ax.set_xlabel(self.xlabel, fontsize=10)

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


        for i in range(len(self.sensitivity[feature])):
            prettyPlot(self.t[feature], self.sensitivity[feature][i],
                       title=feature + ", sensitivity",
                       xlabel=self.xlabel, ylabel="sensitivity",
                       new_figure=False, color=i,
                       nr_hues=len(self.uncertain_parameters),
                       **kwargs)

        plt.ylim([0, 1.05])
        if len(self.sensitivity[feature]) > 4:
            plt.xlim([self.t[feature][0], 1.3*self.t[feature][-1]])

        set_legend(self.listToLatex(self.uncertain_parameters))

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

        xlabels = ["mean", "variance", "$P_5$", "$P_{95}$"]
        xticks = [0, width, distance + width, distance + 2*width]

        values = [self.E[feature], self.Var[feature],
                  self.p_05[feature], self.p_95[feature]]


        ax = prettyBar(values, index=xticks, xlabels=xlabels, ylabel="Value",
                       palette=get_colormap_tableu20())


        if self.sensitivity[feature] is not None:
            pos = 2*distance + 2*width

            ax2 = ax.twinx()

            spines_edge_color(ax2, edges={"top": "None", "bottom": "None",
                                          "right": axis_grey, "left": "None"})
            ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                            color=axis_grey, labelcolor="black", labelsize=labelsize)
            ax2.set_ylabel('sensitivity', fontsize=fontsize)
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

            xticks.append(pos - (i/2. + 0.5)*width)
            xlabels.append("sensitivity")

            location = (0.5, 1.01 + legend_width*0.095)
            lgd = plt.legend(legend_bars,
                             self.listToLatex(self.uncertain_parameters),
                             loc='upper center', bbox_to_anchor=location,
                             ncol=legend_size)
            lgd.get_frame().set_edgecolor(axis_grey)

            fig = plt.gcf()
            fig.subplots_adjust(top=(0.91 - legend_width*0.053))


        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)


        # title = self.filename + ", " + feature
        # title = title.replace("_", "\_")
        plt.suptitle(feature, fontsize=titlesize)

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
        if not self.loaded_flag:
            raise ValueError("Datafile must be loaded")

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data")
    parser.add_argument("-d", "--data_dir", help="Directory the data is stored in", default="data")
    parser.add_argument("-o", "--output_dir", help="Folders to find compare files", default="figures")

    args = parser.parse_args()

    figureformat = ".png"


    plot = PlotUncertainty(data_dir=args.data_dir,
                           output_dir_figures=args.output_dir,
                           figureformat=figureformat)

    # plot.plotAllData()
    plot.plotAllDataFromExploration()

    # sortByParameters(path=output_dir_figures, outputpath=output_dir_figures)
