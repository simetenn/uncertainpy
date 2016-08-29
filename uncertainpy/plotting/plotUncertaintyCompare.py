import os
import argparse
import scipy

import numpy as np
import matplotlib.pyplot as plt

from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.plotting.prettyPlot import prettyPlot, prettyBar
from uncertainpy.plotting.prettyPlot import spines_edge_color, get_current_colormap
# from uncertainpy.plotting.prettyPlot import set_legend, get_colormap_tableu20
from uncertainpy.plotting.prettyPlot import axis_grey, labelsize, fontsize  # , titlesize


# TODO find a good way to find the directory where the data files are


# TODO move load() to it's own class

# TODO compare plots in a grid of all plots,
# such as plotting all features in a grid plot

# TODO Move some parts of the ploting code into it's own class for increased
# readability

# TODO CHange the use of **Kwargs to use a dict for specific plotting commands?

# TODO plot simulator_results

class PlotUncertaintyCompare(PlotUncertainty):


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

            self.loadData(os.path.join(name, filename), create_output_folder=False)

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


    def adaptiveFeatures(self):
        if len(self.features_1d) < 1:
            raise ValueError("No 1D features")


        adaptive_features = []

        for feature in self.features_1d:
            E_prev = self.E_compare[self.compare_folders[0]][feature]
            for data in self.compare_folders[1:]:
                E = self.E_compare[data][feature]
                if E.shape != E_prev.shape:
                    adaptive_features.append(feature)
                    break
                E_prev = E

        return adaptive_features


    def setData(self, dictionary, data, feature):
        for k, d in data.items():
            dictionary[k][feature] = d

    def getData(self, data, feature):

        def _get(dictionary, feature):
            return dictionary[feature]

        result = {}
        for k, d in data.items():
            result[k] = _get(d, feature)

        return result



    def interpolateData(self):
        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")

        self.adaptive_features = self.adaptiveFeatures()

        for feature in self.adaptive_features:
            current_data = self.getData(self.E_compare, feature)
            current_t = self.getData(self.t_compare, feature)

            interpolations = self.createInterpolation(current_data, current_t)
            t, interpolated = self.performInterpolation(current_t, interpolations)

            self.setData(self.E_compare, interpolated, feature)


    def createInterpolation(self, data, t):
        interpolations = {}
        for folder in self.compare_folders:
            if np.all(np.isnan(t[folder])):
                raise AttributeError("Model does not return any t values. Unable to perform interpolation")

            if len(data[folder].shape) == 0:
                raise RuntimeWarning("Data is single values, unable to perform interpolation")
            elif len(data[folder].shape) == 1:
                interpolations[folder] = scipy.interpolate.InterpolatedUnivariateSpline(t[folder], data[folder], k=3)
            else:
                raise NotImplementedError("No support yet for >= 2d interpolation")

        return interpolations



    def performInterpolation(self, t, interpolations):
        lengths = []

        for folder in self.compare_folders:
            lengths.append(len(t[folder]))

        index_max_len = np.argmax(lengths)
        t = t[self.compare_folders[index_max_len]]

        interpolated_solves = {}
        final_t = {}
        for inter in interpolations:
            interpolated_solves[inter] = interpolations[inter](t)
            final_t[inter] = t

        return final_t, interpolated_solves




    def plotCompareMean(self, feature="directComparison",
                        hardcopy=True, show=False, **kwargs):

        self.logger.debug("plotting: {}, mean, compare".format(feature))

        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")


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

        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])
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

        self.logger.debug("plotting: {}, variance, compare".format(feature))

        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")


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

        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])
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

        self.logger.debug("plotting: {}, mean and variance, compare".format(feature))



        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")

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
                                feature + ", mean and variance", "time", "voltage, mean",
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

        self.logger.debug("plotting: {}, confidence-interval, compare".format(feature))

        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")

        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        color = 0
        max_values = []
        min_values = []
        new_figure = True

        for compare in self.compare_folders:
            self.t = self.t_compare[compare]
            self.E = self.E_compare[compare]
            self.p_05 = self.p_05_compare[compare]
            self.p_95 = self.p_95_compare[compare]

            if new_figure:
                ax = prettyPlot(self.t[feature], self.E[feature],
                                title=feature + " ,90\% Confidence interval",
                                xlabel="time", ylabel="voltage", color=color,
                                label=compare.replace("_", " ") + ", Mean", **kwargs)

                colors = get_current_colormap()
            else:
                ax.plot(self.t[feature], self.E[feature],
                        color=colors[color], linewidth=2, antialiased=True,
                        zorder=3, label=compare.replace("_", " ") + ", Mean")

            plt.fill_between(self.t[feature], self.p_05[feature], self.p_95[feature],
                             alpha=0.5, color=colors[color],
                             label=compare.replace("_", " ") + ", 90\% CI",
                             antialiased=True)



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
        plt.legend(ncol=2)

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


        self.logger.debug("plotting: {}, sensitivity, compare".format(feature))

        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")


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
                           title=feature + ", sensitivity",
                           xlabel="time", ylabel="sensitivity",
                           new_figure=new_figure, **kwargs)

                new_figure = False

            plt.legend(legend)
            plt.ylim([0, 1.3])


            if hardcopy:
                plt.savefig(os.path.join(self.full_output_dir_figures,
                                         feature + "_sensitivity_" + parameter_names[i] + "_compare" + self.figureformat),
                            bbox_inches="tight")
                if not show:
                    plt.close()

            if show:
                plt.show()



    def plotCompareAttributeFeature0d(self, feature, attribute="E", attribute_name="mean",
                                      hardcopy=True, show=False, **kwargs):
        self.logger.debug("plotting: {}, {}, compare".format(feature, attribute_name))


        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")

        if feature not in self.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))

        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        width = 0.2
        distance = 0.3

        values = []
        xlabels = []
        xticks = []
        pos = 0

        for compare in self.compare_folders:
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + 0.5*width)
            value = getattr(self, attribute + "_compare")[compare][feature]
            values.append(value)

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



    def plotCompareMeanFeature0d(self, feature, hardcopy=True,
                                 show=False, **kwargs):
        self.plotCompareAttributeFeature0d(feature, attribute="E",
                                           attribute_name="mean",
                                           hardcopy=hardcopy,
                                           show=show, **kwargs)


    def plotCompareVarianceFeature0d(self, feature, hardcopy=True,
                                     show=False, **kwargs):
        self.plotCompareAttributeFeature0d(feature, attribute="Var",
                                           attribute_name="variance",
                                           hardcopy=hardcopy,
                                           show=show, **kwargs)


    def plotCompareConfidenceIntervalFeature0d(self, feature, hardcopy=True,
                                               show=False, **kwargs):

        self.logger.debug("plotting: {}, confidence-interval, compare".format(feature))


        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")

        if feature not in self.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))


        width = 0.2
        distance = 0.5

        min_values = [0]
        max_values = [0]

        values = []
        xlabels = []
        xticks = []
        index = []
        pos = 0

        for compare in self.compare_folders:
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + width)
            index.extend([pos + 0.5*width])

            values.append(self.p_05_compare[compare][feature])
            min_values.append(self.p_05_compare[compare][feature].min())
            max_values.append(self.p_05_compare[compare][feature].max())

            pos += distance + 2*width

        prettyBar(values, index=index, xticks=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=2, color=1, label="$P_{5}$", **kwargs)



        values = []
        xlabels = []
        xticks = []
        index = []
        pos = 0

        for compare in self.compare_folders:
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + width)
            index.extend([pos + 1.5*width])

            values.append(self.p_95_compare[compare][feature])
            min_values.append(self.p_95_compare[compare][feature].min())
            max_values.append(self.p_95_compare[compare][feature].max())

            pos += distance + 2*width

        prettyBar(values, index=index, xticks=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=2, color=0, label="$P_{95}$", new_figure=False, **kwargs)


        plt.legend()

        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])

        plt.title(feature + ", 90 \% confidence interval")

        save_name = feature + "_confidence-interval_compare"

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()





    def plotCompareAttributeFeature1dFractional(self, feature="directComparison", attribute="E",
                                                attribute_name="mean", reference_name="pc",
                                                hardcopy=True, show=False, **kwargs):
        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")


        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))

        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        legend = []
        new_figure = True
        min_values = []
        max_values = []

        compares = self.compare_folders[:]
        compares.remove(reference_name)

        for compare in compares:
            legend.append(compare.replace("_", " "))

            value = getattr(self, attribute + "_compare")
            fractional_difference_mean = self._fractional_difference(value[reference_name][feature],
                                                                     value[compare][feature])
            min_values.append(fractional_difference_mean.min())
            max_values.append(fractional_difference_mean.max())


            title = feature + ", $\\frac{{|{0}_{{{2}}} - {1}_{{{2}}}|}}{{{0}_{{{2}}}}}$".format(reference_name.upper(), compare.split("_")[0].upper(), attribute_name)

            prettyPlot(self.t[feature], fractional_difference_mean,
                       title, "time", "voltage",
                       new_figure=new_figure, nr_hues=len(compares),
                       **kwargs)


            new_figure = False


        save_name = feature + "_" + attribute_name + "_compare_fractional"

        plt.legend(legend)
        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()



    def plotCompareFractionalMean(self, feature="directComparison",
                                  hardcopy=True, show=False, **kwargs):
        self.plotCompareAttributeFeature1dFractional(feature=feature,
                                                     attribute="E",
                                                     attribute_name="mean",
                                                     hardcopy=hardcopy,
                                                     show=show,
                                                     **kwargs)


    def plotCompareFractionalVariance(self, feature="directComparison",
                                      hardcopy=True, show=False, **kwargs):
        self.plotCompareAttributeFeature1dFractional(feature=feature,
                                                     attribute="Var",
                                                     attribute_name="variance",
                                                     hardcopy=hardcopy,
                                                     show=show,
                                                     **kwargs)




    def plotCompareFractionalConfidenceInterval(self, feature="directComparison",
                                                reference_name="pc",
                                                hardcopy=True,
                                                show=False,
                                                **kwargs):

        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")

        if feature not in self.features_1d:
            raise ValueError("%s is not a 1D feature" % (feature))


        new_figure = True
        color = 0

        min_values = []
        max_values = []

        compares = self.compare_folders[:]
        compares.remove(reference_name)

        for compare in compares:
            fractional_difference_mean = self._fractional_difference(self.E_compare[reference_name][feature],
                                                                     self.E_compare[compare][feature])
            fractional_difference_05 = self._fractional_difference(self.p_05_compare[reference_name][feature],
                                                                   self.p_05_compare[compare][feature])
            fractional_difference_95 = self._fractional_difference(self.p_95_compare[reference_name][feature],
                                                                   self.p_95_compare[compare][feature])

            min_values.append(fractional_difference_mean.min())
            max_values.append(fractional_difference_mean.max())
            min_values.append(fractional_difference_05.min())
            max_values.append(fractional_difference_05.max())
            min_values.append(fractional_difference_95.min())
            max_values.append(fractional_difference_95.max())

            title = feature + ", 90\% confidence interval, $\\frac{{|{0} - {1}|}}{{{0}}}$".format(reference_name, compare.split("_")[0])

            if new_figure:
                ax = prettyPlot(self.t[feature], fractional_difference_mean, title=title,
                                xlabel="time", ylabel="voltage", color=color,
                                nr_hues=len(compares), label=compare.replace("_", " ") + ", Mean",
                                **kwargs)

                colors = get_current_colormap()
            else:
                ax.plot(self.t[feature], fractional_difference_mean,
                        color=colors[color], linewidth=2, antialiased=True,
                        zorder=3, label=compare.replace("_", " ") + ", Mean")

            ax.fill_between(self.t[feature], fractional_difference_05,
                            fractional_difference_95,
                            alpha=0.5, color=colors[color], label=compare.replace("_", " ") + ", 90\% CI")


            new_figure = False
            color += 1

        save_name = feature + "_confidence-interval_compare_fractional"

        plt.legend(ncol=2)
        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()



    def _fractional_difference(self, x, y):
        return abs(x - y)/x


    def plotCompareFractionalAttributeFeature0d(self, feature=None,
                                                attribute="E",
                                                attribute_name="mean",
                                                reference_name="pc",
                                                hardcopy=True,
                                                show=False,
                                                **kwargs):

        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")


        if feature not in self.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))


        if attribute not in ["E", "Var"]:
            raise ValueError("{} is not a supported attribute".format(attribute))


        width = 0.2
        distance = 0.3

        values = []
        xlabels = []
        xticks = []
        pos = 0

        compares = self.compare_folders[:]
        compares.remove(reference_name)

        for compare in compares:
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + 0.5*width)
            getattr(self, attribute + "_compare")[compare][feature]
            values.append(self._fractional_difference(getattr(self, attribute + "_compare")[reference_name][feature],
                                                      getattr(self, attribute + "_compare")[compare][feature]))


            pos += distance + width

        prettyBar(values, index=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=len(self.compare_folders), **kwargs)

        title = feature + ", $\\frac{{|{0}_{{{2}}} - {1}_{{{2}}}|}}{{{0}_{{{2}}}}}$".format(reference_name.upper(), compare.split("_")[0].upper(), attribute_name)
        plt.title(title)

        save_name = feature + "_" + attribute_name + "_compare_fractional"

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()


    def plotCompareFractionalMeanFeature0d(self, feature="directComparison",
                                           hardcopy=True, show=False, **kwargs):
        self.plotCompareFractionalAttributeFeature0d(feature=feature,
                                                     attribute="E",
                                                     attribute_name="mean",
                                                     hardcopy=hardcopy,
                                                     show=show,
                                                     **kwargs)


    def plotCompareFractionalVarianceFeature0d(self, feature="directComparison",
                                               hardcopy=True,
                                               show=False,
                                               **kwargs):
        self.plotCompareFractionalAttributeFeature0d(feature=feature,
                                                     attribute="Var",
                                                     attribute_name="variance",
                                                     hardcopy=hardcopy,
                                                     show=show,
                                                     **kwargs)


    def plotCompareFractionalConfidenceIntervalFeature0d(self, feature=None,
                                                         reference_name="pc",
                                                         hardcopy=True,
                                                         show=False,
                                                         **kwargs):
        if not self.loaded_compare_flag:
            raise ValueError("Datafiles must be loaded")


        if feature not in self.features_0d:
            raise ValueError("%s is not a 0D feature" % (feature))


        width = 0.2
        distance = 0.3

        min_values = [0]
        max_values = [0]

        values = []
        xlabels = []
        xticks = []
        index = []
        pos = 0

        compares = self.compare_folders[:]
        compares.remove(reference_name)

        for compare in compares:
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + width)
            index.extend([pos + 0.5*width])

            value = self._fractional_difference(self.p_05_compare[reference_name][feature],
                                                self.p_05_compare[compare][feature])
            values.append(value)

            min_values.append(value.min())
            max_values.append(value.max())

            pos += distance + 2*width

        prettyBar(values, index=index, xticks=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=2, color=1, label="$P_{5}$", **kwargs)

        values = []
        xlabels = []
        xticks = []
        index = []
        pos = 0

        for compare in compares:
            xlabels.append(compare.replace("_", " "))
            xticks.append(pos + width)
            index.extend([pos + 1.5*width])

            value = self._fractional_difference(self.p_95_compare[reference_name][feature],
                                                self.p_95_compare[compare][feature])
            values.append(value)

            min_values.append(value.min())
            max_values.append(value.max())

            pos += distance + 2*width

        prettyBar(values, index=index, xticks=xticks, xlabels=xlabels, ylabel=feature,
                  nr_hues=2, color=0, label="$P_{95}$", new_figure=False, **kwargs)


        plt.legend()

        title = feature + ", $\\frac{{|{0} - {1}|}}{{{0}}}$, 90\\% Confidence interval".format(reference_name.upper(), compare.split("_")[0].upper())
        plt.title(title)

        plt.ylim([min(min_values)*0.99, max(max_values)*1.3])

        save_name = feature + "_confidence-interval_compare_fractional"

        if hardcopy:
            plt.savefig(os.path.join(self.compare_output_dir_figures,
                                     save_name + self.figureformat))
            if not show:
                plt.close()

        if show:
            plt.show()





    def plotCompare1dFeatures(self, hardcopy=True, show=False):
        for feature in self.features_1d:
            self.plotCompareMean(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareVariance(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareMeanAndVariance(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareConfidenceInterval(feature=feature, hardcopy=hardcopy, show=show)



    def plotCompareFractional1dFeatures(self, hardcopy=True, show=False):
        for feature in self.features_1d:
            self.plotCompareFractionalMean(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareFractionalVariance(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareFractionalConfidenceInterval(feature=feature, hardcopy=hardcopy, show=show)



    def plotCompare0dFeatures(self, hardcopy=True, show=False):
        for feature in self.features_0d:
            self.plotCompareMeanFeature0d(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareVarianceFeature0d(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareConfidenceIntervalFeature0d(feature=feature, hardcopy=hardcopy, show=show)



    def plotCompareFractional0dFeatures(self, hardcopy=True, show=False):
        for feature in self.features_0d:
            self.plotCompareFractionalMeanFeature0d(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareFractionalVarianceFeature0d(feature=feature, hardcopy=hardcopy, show=show)
            self.plotCompareFractionalConfidenceIntervalFeature0d(feature=feature, hardcopy=hardcopy, show=show)


    def plotCompareFractional(self, hardcopy=True, show=False):
        self.logger.info("Plotting fractional data")

        self.plotCompareFractional1dFeatures(hardcopy=hardcopy, show=show)
        self.plotCompareFractional0dFeatures(hardcopy=hardcopy, show=show)


    def plotCompare(self, hardcopy=True, show=False):
        self.logger.info("Plotting MC/PC compare data")

        self.plotCompare1dFeatures(hardcopy=hardcopy, show=show)
        self.plotCompare0dFeatures(hardcopy=hardcopy, show=show)


    def plotCompareAll(self, filename, compare_folders,
                       hardcopy=True, show=False):
        self.logger.info("Comparing MC/PC data")

        self.loadCompareData(filename, compare_folders)

        self.plotCompare(hardcopy=hardcopy, show=show)
        self.plotCompareFractional(hardcopy=hardcopy, show=show)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot data")
    parser.add_argument("-d", "--data_dir", help="Directory the data is stored in")

    args = parser.parse_args()

    data_dir = "data/"
    output_dir_figures = "figures/"
    figureformat = ".png"

    if args.data_dir:
        data_dir = "%s/" % os.path.join(data_dir, args.data_dir)
        output_dir_figures = "%s/" % os.path.join(output_dir_figures, args.data_dir)


    plot = PlotUncertaintyCompare(data_dir=data_dir,
                                  output_dir_figures=output_dir_figures,
                                  figureformat=figureformat)

    # plot.plotAllData()
    plot.plotCompareAll()


    # sortByParameters(path=output_dir_figures, outputpath=output_dir_figures)
