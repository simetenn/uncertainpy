# ## TODO
# Test out different types of polynomial chaos methods

# Do dependent variable stuff

# Do a mc analysis after u_hat is generated

# Create a class of this stuff

# Instead of giving results as an average of the response, make it
# feature based. For example, count the number of spikes, and the
# average the number of spikes and time between spikes.

# Make a data seelection process before PC expansion to look at
# specific features. This data selection should be the same as what is
# done for handling spikes from experiments. One example is a low pass
# filter and a high pass filter.

# Use a recursive neural network

# Compare with a pure MC calculation


import time
import datetime
import scipy.interpolate
import os
import sys
import subprocess

import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt

from prettyPlot import prettyPlot
from memory import Memory
from distribution import Distribution
from model import Model
from parameters import Parameters


class UncertaintyEstimations():
    def __init__(self, model, parameters, fitted_parameters, outputdir="figures/"):
        self.UncertaintyEstimations = []

        self.model = model
        self.outputdir = outputdir
        self.parameters = parameters
        self.fitted_parameters = fitted_parameters

        self.t_start = time.time()
        self.memory_report = Memory()

    def initialize(self):
        for distribution_function in distribution_functions:
            print "Running for distribution: " + distribution_function.__name__.split("_")[0]

            for interval in intervals[distribution_function.__name__.lower().split("_")[0]]:
                self.UncertaintyEstimations.append(UncertaintyEstimation(self.model, self.parameters, self.fitted_parameters, self.memory_report))

    def run(self):
        for uncertaintyEstimation in UncertaintyEstimations:
            uncertaintyEstimation.singleParameters()
            uncertaintyEstimation.allParameters()


    def exploreSingleParameters(self, distribution_functions, intervals):
        for distribution_function in distribution_functions:
            print "Running for distribution: " + distribution_function.__name__.split("_")[0]

            for interval in intervals[distribution_function.__name__.lower().split("_")[0]]:
                folder_name = distribution_function.__name__.lower().split("_")[0] + "_" + str(interval)
                current_outputdir = os.path.join(outputdir, folder_name)

                print "Running for interval: %2.4g" % (interval)
                singleParameters(distribution=Distribution(distribution_function, interval),
                                 outputdir=current_outputdir)



class UncertaintyEstimation():
    def __init__(self, model, parameters, outputdir="figures/"):
        """
        model: Model object
        parameters: Parameter object
        distribution_functions: a function that
        outputdir: Where to save the results. Default = "figures/"
        """


        self.outputdir = outputdir
        self.parameters = parameters


        self.figureformat = ".png"
        self.t_start = time.time()

        self.model = model

        self.M = 3
        self.U_hat = None
        self.distribution = None
        self.solves = None
        self.mc_samples = 10**3
        self.t = None
        self.P = None
        self.nodes = None
        self.sensitivity = None

        if not os.path.isdir(self.outputdir):
            os.makedirs(self.outputdir)


    def createPCExpansion(self, parameter_name=None, feature=None):

        if parameter_name is None:
            parameter_space = self.parameters.getIfFitted("parameter_space")
            parameter_name = self.parameters.getIfFitted("name")
        else:
            parameter_space = [self.parameters.get(parameter_name).parameter_space]
            parameter_name = [parameter_name]


        self.distribution = cp.J(*parameter_space)
        self.P = cp.orth_ttr(self.M, self.distribution)
        nodes = self.distribution.sample(2*len(self.P), "M")
        solves = []

        i = 0.
        for s in nodes.T:
            if isinstance(s, float) or isinstance(s, int):
                s = [s]

            sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
            sys.stdout.flush()

            # New setparameters
            tmp_parameters = {}
            j = 0
            for parameter in parameter_name:
                tmp_parameters[parameter] = s[j]
                j += 1

            self.model.saveParameters(tmp_parameters)
            if self.model.run() == -1:
                return -1

            V = np.load("tmp_U.npy")
            t = np.load("tmp_t.npy")

            # Do a feature selection here. Make it so several feature
            # selections are performed at this step. Do this when
            # rewriting it as a class

            if feature is not None:
                V = feature(V)

            interpolation = scipy.interpolate.InterpolatedUnivariateSpline(t, V, k=3)
            solves.append((t, V, interpolation))

            i += 1

        print "\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100)

        solves = np.array(solves)
        lengths = []
        for s in solves[:, 0]:
            lengths.append(len(s))

        index_max_len = np.argmax(lengths)
        self.t = solves[index_max_len, 0]

        interpolated_solves = []
        for inter in solves[:, 2]:
            interpolated_solves.append(inter(self.t))

        self.U_hat = cp.fit_regression(self.P, nodes, interpolated_solves, rule="LS")



    def timePassed(self):
        return time.time() - self.t_start


    def singleParameters(self):
        for fitted_parameter in self.parameters.fitted_parameters:
            print "\rRunning for " + fitted_parameter + "                     "

            if self.createPCExpansion(fitted_parameter) == -1:
                print "Calculations aborted for " + fitted_parameter
                return -1

            try:
                self.E = cp.E(self.U_hat, self.distribution)
                self.Var = cp.Var(self.U_hat, self.distribution)

                self.plotV_t(fitted_parameter)

                samples = self.distribution.sample(self.mc_samples, "M")
                self.U_mc = self.U_hat(samples)

                self.p_10 = np.percentile(self.U_mc, 10, 1)
                self.p_90 = np.percentile(self.U_mc, 90, 1)

                self.plotConfidenceInterval(fitted_parameter + "_confidence_interval")

            except MemoryError:
                print "Memory error, calculations aborted"
                return -1


    def allParameters(self):
        if self.createPCExpansion() == -1:
            print "Calculations aborted for all"
            return -1
        try:
            self.E = cp.E(self.U_hat, self.distribution)
            self.Var = cp.Var(self.U_hat, self.distribution)

            self.plotV_t("all")

            self.sensitivity = cp.Sens_t(self.U_hat, self.distribution)

            self.plotSensitivity()

            samples = self.distribution.sample(self.mc_samples)

            self.U_mc = self.U_hat(*samples)
            self.p_10 = np.percentile(self.U_mc, 10, 1)
            self.p_90 = np.percentile(self.U_mc, 90, 1)

            self.plotConfidenceInterval("all_confidence_interval")

        except MemoryError:
            print "Memory error, calculations aborted"




        def plotV_t(self, filename):
            color1 = 0
            color2 = 8

            prettyPlot(self.t, self.E, "Mean, " + filename, "time", "voltage", color1)
            plt.savefig(os.path.join(self.outputdir, filename + "_mean" + self.figureformat),
                        bbox_inches="tight")

            prettyPlot(self.t, self.Var, "Variance, " + filename, "time", "voltage", color2)
            plt.savefig(os.path.join(self.outputdir, filename + "_variance" + self.figureformat),
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
            plt.savefig(os.path.join(self.outputdir,
                        filename + "_variance_mean" + self.figureformat),
                        bbox_inches="tight")

            plt.close()

        def plotConfidenceInterval(self, filename):

            ax, color = prettyPlot(self.t, self.E, "Confidence interval", "time", "voltage", 0)
            plt.fill_between(self.t, self.p_10, self.p_90, alpha=0.2, facecolor=color[8])
            prettyPlot(self.t, self.p_90, color=8, new_figure=False)
            prettyPlot(self.t, self.p_10, color=9, new_figure=False)
            prettyPlot(self.t, self.E, "Confidence interval", "time", "voltage", 0, False)

            plt.ylim([min([min(self.p_90), min(self.p_10), min(self.E)]),
                      max([max(self.p_90), max(self.p_10), max(self.E)])])

            plt.legend(["Mean", "$P_{90}$", "$P_{10}$"])
            plt.savefig(os.path.join(self.outputdir, filename + self.figureformat),
                        bbox_inches="tight")

            plt.close()

        def plotSensitivity(self):
            parameter_names = self.parameters.get("name")

            for i in range(len(self.sensitivity)):
                prettyPlot(self.t, self.sensitivity[i],
                           parameter_names[i] + " sensitivity", "time",
                           "sensitivity", i, True)
                plt.title(parameter_names[i] + " sensitivity")
                plt.ylim([0, 1.05])
                plt.savefig(os.path.join(self.outputdir,
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
            plt.savefig(os.path.join(self.outputdir, "sensitivity" + self.figureformat),
                        bbox_inches="tight")




if __name__ == "__main__":

    modelfile = "INmodel.hoc"
    modelpath = "neuron_models/dLGN_modelDB/"
    parameterfile = "Parameters.hoc"

    parameters = {
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


    interval = 5*10**-2

    # def normal_function(parameter, interval):
    #     return cp.Normal(parameter, abs(interval*parameter))
    #
    #
    # def uniform_function(parameter, interval):
    #     return cp.Uniform(parameter - abs(interval*parameter),
    #                       parameter + abs(interval*parameter))

    #test_parameters = ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat"]

    distribution_function = Distribution(interval).uniform
    distribution_functions = {"Rm": distribution_function, "Epas": distribution_function}
    test_parameters = ["Rm", "Epas"]

    parameters = Parameters(parameters, distribution_functions, test_parameters)

    model = Model(modelfile, modelpath, parameterfile, parameters)
    test = UncertaintyEstimation(model, parameters, "figures/test")
    test.singleParameters()
    test.allParameters()

    # singleParameters(distribution = Distribution(normal_function, 0.01), outputdir = figurepath + "test_single")

    # allParameters(fitted_parameters = test_parameters, outputdir = figurepath + "test_all")

# allParameters(distribution = Distribution(normal_function, 0.1),
#             fitted_parameters = fitted_parameters, outputdir = figurepath + "test_all")
"""
n_intervals = 10
distributions = [uniform_function, normal_function]
interval_range = {"normal" : np.linspace(10**-4, 10**-1, n_intervals),
                  "uniform" : np.linspace(5*10**-5, 5*10**-2, n_intervals)}
exploreSingleParameters(distributions, interval_range, figurepath + "single")


n_intervals = 10
distributions = [uniform_function, normal_function]
interval_range = {"normal" : np.linspace(10**-3, 10**-1, n_intervals),
                  "uniform" : np.linspace(5*10**-5, 5*10**-2, n_intervals)}
exploreAllParameters(distributions, interval_range, figurepath + "all")
"""

subprocess.Popen(["play", "-q", "ship_bell.wav"])
print "The total runtime is: " + str(datetime.timedelta(seconds=(test.timePassed())))
