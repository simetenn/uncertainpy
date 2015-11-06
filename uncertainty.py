# TODO Test out different types of polynomial chaos methods

# TODO Test correlation


# TODO Do dependent variable stuff

# TODO Do a regression analysis to figure out which variables are dependent on
# each other

# TODO Do a mc analysis after u_hat is generated

# TODO Instead of giving results as an average of the response, make it
# feature based. For example, count the number of spikes, and the
# average the number of spikes and time between spikes.

# TODO Make a data selection process before PC expansion to look at
# specific features. This data selection should be the same as what is
# done for handling spikes from experiments. One example is a low pass
# filter and a high pass filter.

# TODO Use a recursive neural network


# TODO Atm parameter are both in the model object and in the parameter object.
# Make it so they only are one place?

# TODO Can remove the uncertain parameter and instead test if the parameter has
# a distribution function?

# TODO have the option of saving the exploration by parameters instead of by distribution

# TODO WORKING ON THIS!!!!
# TODO remove the plotting from this program, only save the data

# Figures are always saved on the format:
# output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension

# TODO use sumatra when starting runs

# TODO save the entire class to file

import time
import datetime
import scipy.interpolate
import os
import subprocess
import sys
import shutil
import h5py

import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import multiprocess as mp
# import multiprocessing as mp

from xvfbwrapper import Xvfb

# Imported from my own files
from prettyPlot import prettyPlot
from memory import Memory
from distribution import Distribution
from model import Model
from parameters import Parameters
from plotUncertainty import PlotUncertainty
from collect_by_parameter import sortByParameters
# from evaluateNodeFunction import evaluateNodeFunction



class UncertaintyEstimations():
    def __init__(self, model, uncertain_parameters, distributions,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 output_dir_data="data/",
                 supress_output=True,
                 CPUs=mp.cpu_count(),
                 interpolate_union=False,
                 rosenblatt=False):

        # Figures are always saved on the format:
        # output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-format

        self.UncertaintyEstimations = []

        self.model = model
        self.uncertain_parameters = uncertain_parameters
        self.distributions = distributions
        self.output_dir_figures = output_dir_figures
        self.output_dir_data = output_dir_data

        self.supress_output = supress_output
        self.CPUs = CPUs
        self.interpolate_union = interpolate_union
        self.rosenblatt = rosenblatt
        self.figureformat = figureformat

        self.initialize()

        self.t_start = time.time()



    def initialize(self):
        for distribution_function in self.distributions:
            for interval in self.distributions[distribution_function]:
                # TODO update this when figured out the saving stuff
                current_output_dir_figures = os.path.join(self.output_dir_figures,
                                                          distribution_function + "_%g" % interval)
                distribution = getattr(Distribution(interval), distribution_function)
                parameters = Parameters(self.model.parameters, distribution,
                                        self.uncertain_parameters)
                self.UncertaintyEstimations.append(UncertaintyEstimation(self.model, parameters,
                                                                         output_dir_figures=current_output_dir_figures,
                                                                         figureformat=self.figureformat,
                                                                         output_dir_data=self.output_dir_data,
                                                                         output_data_name=distribution_function + "_%g" % interval,
                                                                         supress_output=self.supress_output,
                                                                         CPUs=self.CPUs,
                                                                         interpolate_union=self.interpolate_union,
                                                                         rosenblatt=self.rosenblatt))

    def exploreParameters(self):
        for uncertaintyEstimation in self.UncertaintyEstimations:
            distribution, interval = uncertaintyEstimation.output_dir_figures.split("/")[-1].split("_")
            print "Running for: " + distribution + " " + interval

            uncertaintyEstimation.singleParameters()
            uncertaintyEstimation.allParameters()



    def timePassed(self):
        return time.time() - self.t_start



class UncertaintyEstimation():
    def __init__(self, model, parameters,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 save_data=True,
                 output_dir_data="data/",
                 output_data_name="uncertainty",
                 supress_output=True,
                 CPUs=mp.cpu_count(),
                 interpolate_union=False,
                 rosenblatt=False):
        """
        model: Model object
        parameters: Parameter object
        output_dir_figures: Where to save the results. Default = "figures/"

        Figures are always saved on the format:
        output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-format
        """

        self.output_dir_figures = output_dir_figures
        self.figureformat = figureformat
        self.save_data = save_data
        self.output_file = os.path.join(output_dir_data, output_data_name)

        self.parameters = parameters

        self.supress_output = supress_output
        self.CPUs = CPUs

        self.model = model

        self.interpolate_union = interpolate_union
        self.rosenblatt = rosenblatt

        self.features = []
        self.M = 3
        self.nr_mc_samples = 10**3


        self.parameter_names = None
        self.parameter_space = None

        self.U_hat = None
        self.distribution = None
        self.solves = None
        self.t = None
        self.E = None
        self.Var = None
        self.p_05 = None
        self.p_95 = None
        self.sensitivity = None
        self.Corr = None
        self.P = None
        self.nodes = None


        if os.path.isdir(self.output_dir_figures):
            shutil.rmtree(self.output_dir_figures)
        os.makedirs(self.output_dir_figures)

        if self.save_data:
            if not os.path.isdir(output_dir_data):
                os.makedirs(output_dir_data)
            else:
                if os.path.isfile(self.output_file):
                    os.remove(self.output_file)

        self.t_start = time.time()


    def __del__(self):
        "delete"


    def toList(self):
        data = []
        for node in self.nodes.T:
            data.append((node, self.tmp_parameter_name, modelfile, modelpath, self.features))

        return data


    def evaluateNode(self, node):
        if isinstance(node, float) or isinstance(node, int):
                node = [node]

        # New setparameters
        tmp_parameters = {}
        j = 0
        for parameter in self.tmp_parameter_name:
            tmp_parameters[parameter] = node[j]
            j += 1

        t, V = self.model.run(tmp_parameters)

        # Do a feature selection here. Make it so several feature
        # selections are performed at this step.
        for feature in self.features:
            V = feature(V)

        interpolation = scipy.interpolate.InterpolatedUnivariateSpline(t, V, k=3)

        return (t, V, interpolation)


    def createPCExpansion(self, parameter_name=None):

        # TODO find a good way to solve the parameter_name poblem
        if parameter_name is None:
            parameter_space = self.parameters.getUncertain("parameter_space")
            self.tmp_parameter_name = self.parameters.getUncertain("name")
        else:
            parameter_space = [self.parameters.get(parameter_name).parameter_space]
            self.tmp_parameter_name = [parameter_name]



        if self.rosenblatt:
            # Create the Multivariat normal distribution
            dist_MvNormal = []
            for parameter in self.parameters.getUncertain("value"):
                dist_MvNormal.append(cp.Normal())

            dist_MvNormal = cp.J(*dist_MvNormal)

            self.distribution = cp.J(*parameter_space)
            #self.P = cp.orth_ttr(self.M, self.distribution)
            self.P = cp.orth_ttr(self.M, dist_MvNormal)

            nodes_MvNormal = dist_MvNormal.sample(2*len(self.P)+1, "M")
            # nodes_MvNormal, weights_MvNormal = cp.generate_quadrature(3, dist_MvNormal,
            #                                                           rule="J", sparse=True)

            nodes = self.distribution.inv(dist_MvNormal.fwd(nodes_MvNormal))
            # weights = weights_MvNormal*self.distribution.pdf(nodes)/dist_MvNormal.pdf(nodes_MvNormal)

            self.distribution = dist_MvNormal

        else:
            self.distribution = cp.J(*parameter_space)
            self.P = cp.orth_ttr(self.M, self.distribution)

            nodes = self.distribution.sample(2*len(self.P)+1, "M")
            # nodes, weights = cp.generate_quadrature(3, self.distribution, rule="J", sparse=True)


        if self.supress_output:
            vdisplay = Xvfb()
            vdisplay.start()

        solves = []
        if self.CPUs > 0:
            pool = mp.Pool(processes=self.CPUs)
            solves = pool.map(self.evaluateNode, nodes.T)
            # solves = pool.map(evaluateNodeFunction, self.toList())

        else:
            for node in nodes.T:
                solves.append(self.evaluateNode(node))

        if self.supress_output:
            vdisplay.stop()

        solves = np.array(solves)

        # Use union to work on all time values when interpolation.
        # If not use the t maximum amount of t values
        if self.interpolate_union:
            i = 0
            tmp_t = solves[0, 0]
            while i < len(solves) - 1:
                tmp_t = np.union1d(tmp_t, solves[i+1, 0])
                i += 1

            self.t = tmp_t
        else:
            lengths = []
            for s in solves[:, 0]:
                lengths.append(len(s))

            index_max_len = np.argmax(lengths)
            self.t = solves[index_max_len, 0]


        interpolated_solves = []
        for inter in solves[:, 2]:
            interpolated_solves.append(inter(self.t))



        if self.rosenblatt:
            #self.U_hat = cp.fit_quadrature(self.P, nodes_MvNormal, weights, interpolated_solves)
            self.U_hat = cp.fit_regression(self.P, nodes_MvNormal, interpolated_solves, rule="T")
        else:
            #self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
            self.U_hat = cp.fit_regression(self.P, nodes, interpolated_solves, rule="T")


    def MC(self, parameter_name=None):
        if parameter_name is None:
            parameter_space = self.parameters.getUncertain("parameter_space")
            self.tmp_parameter_name = self.parameters.getUncertain("name")
        else:
            parameter_space = [self.parameters.get(parameter_name).parameter_space]
            self.tmp_parameter_name = [parameter_name]

        self.distribution = cp.J(*parameter_space)
        samples = self.distribution.sample(self.nr_mc_samples, "M")

        if self.supress_output:
            vdisplay = Xvfb()
            vdisplay.start()

        solves = []

        self.CPUs = 0
        if self.CPUs > 0:
            pool = mp.Pool(processes=self.CPUs)
            solves = pool.map(self.evaluateNode, samples.T)

        else:
            for sample in samples.T:
                solves.append(self.evaluateNode(sample))

        if self.supress_output:
            vdisplay.stop()

        solves = np.array(solves)
        lengths = []
        for s in solves[:, 0]:
            lengths.append(len(s))

        index_max_len = np.argmax(lengths)
        self.t = solves[index_max_len, 0]

        #self.t = np.linspace(solves[0,0], solves[0,0])

        interpolated_solves = []
        for inter in solves[:, 2]:
            interpolated_solves.append(inter(self.t))

        self.E = (np.sum(interpolated_solves, 0).T/self.nr_mc_samples).T
        self.Var = (np.sum((interpolated_solves - self.E)**2, 0).T/self.nr_mc_samples).T

        self.plotV_t("MC")


    def timePassed(self):
        return time.time() - self.t_start


    def singleParameters(self):
        for uncertain_parameter in self.parameters.uncertain_parameters:
            print "\rRunning for " + uncertain_parameter + "                     "

            if self.createPCExpansion(uncertain_parameter) == -1:
                print "Calculations aborted for " + uncertain_parameter
                return -1

            try:
                self.E = cp.E(self.U_hat, self.distribution)
                self.Var = cp.Var(self.U_hat, self.distribution)

                samples = self.distribution.sample(self.nr_mc_samples, "M")
                self.U_mc = self.U_hat(samples)

                self.p_05 = np.percentile(self.U_mc, 5, 1)
                self.p_95 = np.percentile(self.U_mc, 95, 1)

                self.save(uncertain_parameter)

                self.plotV_t(uncertain_parameter)
                self.plotConfidenceInterval(uncertain_parameter + "_confidence-interval")


            except MemoryError:
                print "Memory error, calculations aborted"
                return -1


    def allParameters(self):
        if len(self.parameters.uncertain_parameters) <= 1:
            print "Only 1 uncertain parameter"
            if self.U_hat is None:
                print "Running singleParameters() instead"
                self.singleParameters()
            return

        if self.createPCExpansion() == -1:
            print "Calculations aborted for all"
            return -1
        try:
            self.E = cp.E(self.U_hat, self.distribution)
            self.Var = cp.Var(self.U_hat, self.distribution)

            self.sensitivity = cp.Sens_t(self.U_hat, self.distribution)


            samples = self.distribution.sample(self.nr_mc_samples, "M")

            self.U_mc = self.U_hat(*samples)
            self.p_05 = np.percentile(self.U_mc, 5, 1)
            self.p_95 = np.percentile(self.U_mc, 95, 1)

            self.save("all")

            self.plotV_t("all")
            self.plotSensitivity()
            self.plotConfidenceInterval("all_confidence-interval")



            self.Corr = cp.Corr(self.P, self.distribution)

        except MemoryError:
            print "Memory error: calculations aborted"


    def plotV_t(self, filename):
        color1 = 0
        color2 = 8

        prettyPlot(self.t, self.E, "Mean, " + filename, "time", "voltage", color1)
        plt.savefig(os.path.join(self.output_dir_figures, filename + "_mean" + self.figureformat),
                    bbox_inches="tight")

        prettyPlot(self.t, self.Var, "Variance, " + filename, "time", "voltage", color2)
        plt.savefig(os.path.join(self.output_dir_figures, filename + "_variance" + self.figureformat),
                    bbox_inches="tight")

        ax, tableau20 = prettyPlot(self.t, self.E, "Mean and variance, " + filename,
                                   "time", "voltage, mean", color1)
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on",
                        color=tableau20[color2], labelcolor=tableau20[color2], labelsize=14)
        ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
        ax.spines["right"].set_edgecolor(tableau20[color2])

        ax2.set_xlim([min(self.t), max(self.t)])
        ax2.set_ylim([min(self.Var), max(self.Var)])

        ax2.plot(self.t, self.Var, color=tableau20[color2], linewidth=2, antialiased=True)

        ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
        ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
        ax.spines["left"].set_edgecolor(tableau20[color1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir_figures,
                    filename + "_variance-mean" + self.figureformat),
                    bbox_inches="tight")

        plt.close()

    def plotConfidenceInterval(self, filename):

        ax, color = prettyPlot(self.t, self.E, "Confidence interval", "time", "voltage", 0)
        plt.fill_between(self.t, self.p_05, self.p_95, alpha=0.2, facecolor=color[8])
        prettyPlot(self.t, self.p_95, color=8, new_figure=False)
        prettyPlot(self.t, self.p_05, color=9, new_figure=False)
        prettyPlot(self.t, self.E, "Confidence interval", "time", "voltage", 0, False)

        plt.ylim([min([min(self.p_95), min(self.p_05), min(self.E)]),
                  max([max(self.p_95), max(self.p_05), max(self.E)])])

        plt.legend(["Mean", "$P_{95}$", "$P_{5}$"])
        plt.savefig(os.path.join(self.output_dir_figures, filename + self.figureformat),
                    bbox_inches="tight")

        plt.close()

    def plotSensitivity(self):
        parameter_names = self.parameters.getUncertain("name")

        for i in range(len(self.sensitivity)):
            prettyPlot(self.t, self.sensitivity[i],
                       parameter_names[i] + " sensitivity", "time",
                       "sensitivity", i, True)
            plt.title(parameter_names[i] + " sensitivity")
            plt.ylim([0, 1.05])
            plt.savefig(os.path.join(self.output_dir_figures,
                                     parameter_names[i] +
                                     "_sensitivity" + self.figureformat),
                        bbox_inches="tight")
        plt.close()

        for i in range(len(self.sensitivity)):
            prettyPlot(self.t, self.sensitivity[i], "sensitivity", "time",
                       "sensitivity", i, False)

        plt.ylim([0, 1.05])
        plt.xlim([self.t[0], 1.3*self.t[-1]])
        plt.legend(parameter_names)
        plt.savefig(os.path.join(self.output_dir_figures,
                                 "all_sensitivity" + self.figureformat),
                    bbox_inches="tight")


    def save(self, group_name="default"):
        ### TODO expand the save funcition to also save parameters and model informationty

        f = h5py.File(self.output_file, 'a')

        if "name" not in f.attrs.keys():
            f.attrs["name"] = self.output_file.split("/")[-1]
        if "Uncertain parameters" not in f.attrs.keys():
            f.attrs["Uncertain parameters"] = self.parameters.getUncertain("name")

        if group_name in f.keys():
            del f[group_name]
        group = f.create_group(group_name)

        if self.t is not None:
            group.create_dataset("t", data=self.t)
        if self.E is not None:
            group.create_dataset("E", data=self.E)
        if self.Var is not None:
            group.create_dataset("Var", data=self.Var)
        if self.sensitivity is not None:
            group.create_dataset("sensitivity", data=self.sensitivity)
        if self.p_05 is not None:
            group.create_dataset("p_05", data=self.p_05)
        if self.p_95 is not None:
            group.create_dataset("p_95", data=self.p_95)

        f.close()

if __name__ == "__main__":

    modelfile = "INmodel.hoc"
    modelpath = "neuron_models/dLGN_modelDB/"
    parameterfile = "Parameters.hoc"

    data_dir = "data/"
    output_figures_dir = "figures/"
    figureformat = ".png"
    output_gif_dir = "gifs/"

    original_parameters = {
        "rall": 113,       # Taken from litterature
        "cap": 1.1,        #
        "Rm": 22000,       # Estimated by hand
        "Vrest": -63,      # Experimentally measured
        "Epas": -67,       # Estimated by hand
        "gna": 0.09,
        "nash": -52.6,
        "gkdr": 0.37,
        "kdrsh": -51.2,
        "gahp": 6.4e-5,
        "gcat": 1.17e-5,    # Estimated by hand
        "gcal": 0.0009,
        "ghbar": 0.00011,   # Estimated by hand
        "catau": 50,
        "gcanbar": 2e-8
    }


    fitted_parameters = ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat", "gcal",
                         "ghbar", "catau", "gcanbar"]



    test_parameters = ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat"]

    distribution_function = Distribution(0.1).uniform
    distribution_functions = {"Rm": distribution_function, "Epas": distribution_function}

    test_parameters = "Rm"
    #test_parameters = ["Rm", "Epas", "kdrsh", "catau"]
    #test_parameters = ["gcat", "gcal",
    #                   "ghbar", "gcanbar"]
    test_parameters = ["Rm", "Epas", "kdrsh"]

    memory_report = Memory()
    parameters = Parameters(original_parameters, distribution_function, test_parameters)
    #parameters = Parameters(original_parameters, distribution_function, fitted_parameters)

    model = Model(modelfile, modelpath, parameterfile, original_parameters, memory_report)
    #test = UncertaintyEstimation(model, parameters, "figures/test", CPUs=mp.cpu_count()-1)
    #test.MC()
    # t1 = test.timePassed()
    # test.pseudoMC("Rm")
    # test.plotV_t("MC")
    # t2 = test.timePassed()
    # test.singleParameters()
    # t3 = test.timePassed()
    # print "MC ", t2-t1
    # print "PC ", t3-t2

    #test.singleParameters()
#    test.allParameters()

    test_distributions = {"uniform": [0.04, 0.05, 0.06]}
    #test_distributions = {"uniform": np.linspace(0.01, 0.1, 2)}
    exploration = UncertaintyEstimations(model, test_parameters, test_distributions)
    exploration.exploreParameters()
    #distributions = {"uniform": np.linspace(0.01, 0.1, 10), "normal": np.linspace(0.01, 0.1, 10)}
    #exploration = UncertaintyEstimations(model, fitted_parameters, distributions)
    #exploration.exploreParameters()

    plot = PlotUncertainty(data_dir=data_dir,
                           output_figures_dir=output_figures_dir,
                           figureformat=figureformat,
                           output_gif_dir=output_gif_dir)

    plot.allData()
    plot.gif()
    sortByParameters()


    subprocess.Popen(["play", "-q", "ship_bell.wav"])
    print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
