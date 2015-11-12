# TODO Test out different types of polynomial chaos methods

# TODO Test correlation

# TODO to many npy files are created

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

# Figures are always saved on the format:
# output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-extension

# TODO use sumatra when starting runs

# TODO save the entire class to file

# TODO Add suport for canceling a simulation when it is above a certain memory theeshold

# TODO Pool is created to many times
# TODO xvbf is taking alot of time to start and stop, fix this

import time
import datetime
import scipy.interpolate
import os
import subprocess
import shutil
import h5py

import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import multiprocess as mp
#import multiprocessing as mping

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


def empty(node):
    pass


class UncertaintyEstimations():
    def __init__(self, model, uncertain_parameters, distributions,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 output_dir_data="data/",
                 supress_output=True,
                 CPUs=mp.cpu_count(),
                 interpolate_union=False,
                 rosenblatt=False,
                 memory_log=False):

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
        self.memory_log = memory_log


        self.t_start = time.time()


    def exploreParameters(self):
        for distribution_function in self.distributions:
            for interval in self.distributions[distribution_function]:
                # TODO update this when figured out the saving stuff
                current_output_dir_figures = os.path.join(self.output_dir_figures,
                                                          distribution_function + "_%g" % interval)
                distribution = getattr(Distribution(interval), distribution_function)
                parameters = Parameters(self.model.parameters, distribution,
                                        self.uncertain_parameters)

                print "Running for: " + distribution_function + " " + str(interval)
                uncertainty_estimation = UncertaintyEstimation(self.model, parameters,
                                                               output_dir_figures=current_output_dir_figures,
                                                               figureformat=self.figureformat,
                                                               output_dir_data=self.output_dir_data,
                                                               output_data_name=distribution_function + "_%g" % interval,
                                                               supress_output=self.supress_output,
                                                               CPUs=self.CPUs,
                                                               interpolate_union=self.interpolate_union,
                                                               rosenblatt=self.rosenblatt)

                uncertainty_estimation.singleParameters()
                uncertainty_estimation.allParameters()
                #uncertainty_estimation.noCalculations()
                del uncertainty_estimation

    def timePassed(self):
        return time.time() - self.t_start



class UncertaintyEstimation():
    def __init__(self, model, parameters,
                 save_figures=False,
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

        self.save_figures = save_figures
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


        self.resetValues()


        if self.supress_output:

            self.vdisplay = Xvfb()
            self.vdisplay.start()

        # if self.CPUs > 1:
        #     self.pool = mp.Pool(processes=self.CPUs)


        if save_data:
            if not os.path.isdir(output_dir_data):
                os.makedirs(output_dir_data)
            else:
                if os.path.isfile(self.output_file):
                    os.remove(self.output_file)

        self.t_start = time.time()


    def __del__(self):
        if self.CPUs > 1 and hasattr(self, 'pool'):
            self.pool.close()

        if self.supress_output:
            self.vdisplay.stop()


    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict


    def __setstate__(self, state):
        self.__dict__.update(state)


    def resetValues(self):
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


    def createPCExpansion(self, parameter_name="all"):

        # TODO find a good way to solve the parameter_name poblem
        if parameter_name is "all":
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
        if self.CPUs > 1:
            try:
                self.pool = mp.Pool(processes=self.CPUs)
                solves = self.pool.map(self.evaluateNode, nodes.T)
                # solves = pool.map(evaluateNodeFunction, self.toList())
            except MemoryError:
                return -1
        else:
            try:
                for node in nodes.T:
                    solves.append(self.evaluateNode(node))
            except MemoryError:
                return -1

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


    def MC(self, parameter_name="all"):
        if parameter_name is "all":
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

        #self.CPUs = 0
        if self.CPUs > 0:
            #pool = mp.Pool(processes=self.CPUs)
            solves = self.pool.map(self.evaluateNode, samples.T)

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
        self.Var[parameter_name] = (np.sum((interpolated_solves - self.E)**2, 0).T/self.nr_mc_samples).T

        #self.plotV_t("MC")


    def timePassed(self):
        return time.time() - self.t_start


    def singleParameters(self):
        for uncertain_parameter in self.parameters.uncertain_parameters:
            print "\rRunning for " + uncertain_parameter + "                     "

            self.resetValues()

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

                if self.save_data:
                    self.save(uncertain_parameter)

                if self.save_figures:
                    self.plotAll(uncertain_parameter)

                # self.plotV_t(uncertain_parameter)
                # self.plotConfidenceInterval(uncertain_parameter + "_confidence-interval")


            except MemoryError:
                print "Memory error, calculations aborted"
                return -1

    def allParameters(self):
        if len(self.parameters.uncertain_parameters) <= 1:
            print "Only 1 uncertain parameter"
            # if self.U_hat is None:
            #     print "Running singleParameters() instead"
            #self.singleParameters()
            return

        print "\rRunning for all                     "
        if self.createPCExpansion() == -1:
            print "Calculations aborted for all"
            return -1
        try:
            self.E = cp.E(self.U_hat, self.distribution)
            self.Var = cp.Var(self.U_hat, self.distribution)

            self.sensitivity = cp.Sens_t(self.U_hat, self.distribution)
            self.sensitivityRanking()

            samples = self.distribution.sample(self.nr_mc_samples, "M")

            self.U_mc = self.U_hat(*samples)
            self.p_05 = np.percentile(self.U_mc, 5, 1)
            self.p_95 = np.percentile(self.U_mc, 95, 1)

            if self.save_data:
                self.save("all")

            if self.save_figures:
                self.plotAll("all")

            #self.Corr = cp.Corr(self.P, self.distribution)

        except MemoryError:
            print "Memory error: calculations aborted"


    def noCalculations(self):
        for uncertain_parameter in self.parameters.uncertain_parameters:
            print "No calculations"
            parameter_space = [self.parameters.get(uncertain_parameter).parameter_space]
            #
            self.distribution = cp.J(*parameter_space)
            self.P = cp.orth_ttr(self.M, self.distribution)

            nodes = self.distribution.sample(2*len(self.P)+1, "M")

            #
            # if self.supress_output:
            #     vdisplay = Xvfb()
            #     vdisplay.start()
            #
            # self.pool = mping.Pool(processes=self.CPUs)
            # pool.map(self.model.runNoCalculation(), nodes.T)
            self.pool.map(empty, nodes.T)
            #
            #
            #
            # if self.supress_output:
            #     vdisplay.stop()

    # def evaluateNodeNoCalculation(self, node):
    #     self.model.runNoCalculation()

    def plotAll(self, parameter="all"):
        if os.path.isdir(self.output_dir_figures):
            shutil.rmtree(self.output_dir_figures)
        os.makedirs(self.output_dir_figures)

        self.plotV_t(parameter)
        self.plotConfidenceInterval(parameter)

        if parameter == "all":
            self.plotSensitivity()

    def sensitivityRanking(self):
        self.sensitivity_ranking = {}
        i = 0
        for parameter in self.parameters.getUncertain("name"):
            self.sensitivity_ranking[parameter] = sum(self.sensitivity[i])
            i += 1

        total_sensitivity = 0
        for parameter in self.sensitivity_ranking:
            total_sensitivity += self.sensitivity_ranking[parameter]
        for parameter in self.sensitivity_ranking:
            self.sensitivity_ranking[parameter] /= total_sensitivity


    def plotV_t(self, parameter):
        if parameter not in self.E:
            print "WARNING: %s have not been calculated" % (parameter)
            return

        color1 = 0
        color2 = 8

        prettyPlot(self.t, self.E, "Mean, " + parameter, "time", "voltage", color1)
        plt.savefig(os.path.join(self.output_dir_figures, parameter + "_mean" + self.figureformat),
                    bbox_inches="tight")

        prettyPlot(self.t, self.Var, "Variance, " + parameter, "time", "voltage", color2)
        plt.savefig(os.path.join(self.output_dir_figures, parameter + "_variance" + self.figureformat),
                    bbox_inches="tight")

        ax, tableau20 = prettyPlot(self.t, self.E, "Mean and variance, " + parameter,
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
                    parameter + "_variance-mean" + self.figureformat),
                    bbox_inches="tight")

        plt.close()

    def plotConfidenceInterval(self, parameter):
        if parameter not in self.E:
            print "WARNING: %s have not been calculated" % (parameter)
            return

        ax, color = prettyPlot(self.t, self.E, "Confidence interval", "time", "voltage", 0)
        plt.fill_between(self.t, self.p_05, self.p_95, alpha=0.2, facecolor=color[8])
        prettyPlot(self.t, self.p_95, color=8, new_figure=False)
        prettyPlot(self.t, self.p_05, color=9, new_figure=False)
        prettyPlot(self.t, self.E, "Confidence interval", "time", "voltage", 0, False)

        plt.ylim([min([min(self.p_95), min(self.p_05), min(self.E)]),
                  max([max(self.p_95), max(self.p_05), max(self.E)])])

        plt.legend(["Mean", "$P_{95}$", "$P_{5}$"])
        plt.savefig(os.path.join(self.output_dir_figures, parameter + "_confidence-interval" + self.figureformat),
                    bbox_inches="tight")

        plt.close()

    def plotSensitivity(self):
        if not self.sensitivity:
            print "WARNING: Sensitivity have not been calculated"
            return

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


    def saveAll(self):
        self.save()
        for parameter in self.parameters.getUncertain("name"):
            self.save(parameter)

    def save(self, parameter="all"):
        ### TODO expand the save funcition to also save parameters and model informationty

        f = h5py.File(self.output_file, 'a')

        if "name" not in f.attrs.keys():
            f.attrs["name"] = self.output_file.split("/")[-1]
        if "Uncertain parameters" not in f.attrs.keys():
            f.attrs["Uncertain parameters"] = self.parameters.getUncertain("name")



        if parameter in f.keys():
            del f[parameter]
        group = f.create_group(parameter)

        if parameter in self.t:
            group.create_dataset("t", data=self.t)
        if parameter in self.E:
            group.create_dataset("E", data=self.E)
        if parameter in self.Var:
            group.create_dataset("Var", data=self.Var)
        if parameter in self.p_05:
            group.create_dataset("p_05", data=self.p_05)
        if parameter in self.p_95:
            group.create_dataset("p_95", data=self.p_95)


        # TODO check if this saves correctly
        if self.sensitivity is not None and parameter == "all":
            group.create_dataset("sensitivity", data=self.sensitivity)

            i = 0
            for parameter in self.parameters.getUncertain("name"):
                if parameter not in f.keys():
                    f.create_group(parameter)

                f[parameter].create_dataset("total sensitivity", data=self.sensitivity_ranking[parameter])
                f[parameter].create_dataset("sensitivity", data=self.sensitivity[i])
                i += 1


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

    memory = Memory(1)
    memory.start()

    fitted_parameters = ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat", "gcal",
                         "ghbar", "catau", "gcanbar"]

    test_parameters = ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat"]
    test_parameters = ["Rm", "Epas"]

    distribution_function = Distribution(0.1).uniform
    distribution_functions = {"Rm": distribution_function, "Epas": distribution_function}



    #parameters = Parameters(original_parameters, distribution_function, test_parameters)
    parameters = Parameters(original_parameters, distribution_function, fitted_parameters)

    model = Model(modelfile, modelpath, parameterfile, original_parameters)

    test_distributions = {"uniform": [0.05, 0.06], "normal": [0.04, 0.05]}
    #test_distributions = {"uniform": np.linspace(0.01, 0.1, 2)}



    #
    percentages = np.linspace(0.01, 0.1, 9)
    test_distributions = {"uniform": percentages}
    exploration = UncertaintyEstimations(model, test_parameters, test_distributions)
    exploration.exploreParameters()

    #distributions = {"uniform": np.linspace(0.01, 0.1, 10), "normal": np.linspace(0.01, 0.1, 10)}
    #percentages = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    # # percentages = [0.02, 0.03, 0.04, 0.05]
    # distributions = {"uniform": percentages}
    # exploration = UncertaintyEstimations(model, fitted_parameters, distributions)
    # exploration.exploreParameters()

    # distributions = {"normal": percentages}
    # exploration = UncertaintyEstimations(model, fitted_parameters, distributions)
    # exploration.exploreParameters()
    memory.end()



    # plot = PlotUncertainty(data_dir=data_dir,
    #                        output_figures_dir=output_figures_dir,
    #                        figureformat=figureformat,
    #                        output_gif_dir=output_gif_dir)
    #
    # plot.allData()
    # plot.gif()
    # sortByParameters()


    subprocess.Popen(["play", "-q", "ship_bell.wav"])
    print "The total runtime is: " + str(datetime.timedelta(seconds=(exploration.timePassed())))
