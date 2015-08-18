### TODO
# Test out different types of polynomial chaos methods

# Do dependent variable stuff

# Do a mc analysis after u_hat is generated

# Create a class of this stuff

# Instead of giving results as an average of the response, make it
# feature based. For example, count the number of spikes, and the
# average the number of spikes and time between spikes.

# Make a data selection process before PC expansion to look at
# specific features. This data selection should be the same as what is
# done for handling spikes from experiments. One example is a low pass
# filter and a high pass filter.

# Use a recursive neural network

import time, subprocess, datetime, cPickle, scipy.interpolate, os, sys, string
import numpy as np
import scipy as scp
import chaospy as cp
import matplotlib.pyplot as plt

from xvfbwrapper import Xvfb

from prettyPlot import prettyPlot
from memory import Memory


# Global parameters
interval = 5*10**-4
nr_mc_samples = 10**3

modelfile = "INmodel.hoc"
modelpath = "neuron_models/dLGN_modelDB/"


M = 3

parameters = {
    "rall" : 113,       # Taken from litterature 
    "cap" : 1.1,        # 
    "Rm" : 22000,       # Estimated by hand
    "Vrest" : -63,      # Experimentally measured
    "Epas" : -67,       # Estimated by hand
    "gna" :  0.09,      
    "nash" : -52.6,
    "gkdr" : 0.37,
     "kdrsh" : -51.2,
     "gahp" : 6.4e-5,
     "gcat" :1.17e-5,    # Estimated by hand
     "gcal" :0.0009,
     "ghbar" :0.00011,   # Estimated by hand
     "catau" : 50,
     "gcanbar" : 2e-8
}


fitted_parameters =   ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat",
                        "gcal", "ghbar", "catau", "gcanbar"]



class UncertaintyEstimation():
    def __init__(self, modelfile, modelpath, parameterfile, parameters, fitted_parameters, outputdir = "figures/"):
        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(filepath)
        self.outputdir = outputdir
        self.parameters = parameters
        self.fitted_parameters = fitted_parameters
        self.figureformat = ".png"
        
        self.memory_threshold = 90
        self.delta_poll = 1

        self.memory_report = Memory()
        self.t_start = time.time()
        
        self.model = Model(self.modelfile, self.modelpath, self.parameterfile, self.filedir)

        
#        def normal_function(parameter, interval):
#            return cp.Normal(parameter, abs(interval*parameter))
#
#    
#        def uniform_function(parameter, interval):
#            return cp.Uniform(parameter - abs(interval*parameter),
#                              parameter + abs(interval*parameter))
#
#
#        self.Distribution = Distribution(interval)
#        #self.normal = Distribution(normal_function, interval)
#        #self.uniform = Distribution(uniform_function, interval)
#        self.normal = self.Distribution.normal
#        self.uniform = self.Distribution.uniform

        self.parameter_space = None

        self.cvode_active = True
        self.M = 3
        self.U_hat = None
        self.dist = None
        self.solves = None
        self.t_max = None
        self.P = None
        self.nodes = None
        self.sensitivity = None


    def newParameterSpace(self, distribution):
        """
        Generalized parameter space creation
        """

        parameter_space = {}

        if type(self.fitted_parameters) == str:
            parameter_space[self.fitted_parameters] = distribution(self.parameters[self.fitted_parameters])
            self.parameter_space = parameter_space
        else:
            for param in self.fitted_parameters:
                parameter_space[self.param] = distribution(self.parameters[param])

            self.parameter_space = parameter_space
            



    def createPCExpansion(feature = None, cvode_active=True):

        self.dist = cp.J(*self.parameter_space.values())
        self.P = cp.orth_ttr(self.M, self.dist)
        self.nodes = self.dist.sample(2*len(P), "M")
        self.solves = []



        i = 0.
        for s in nodes.T:
            if isinstance(s, float) or isinstance(s, int):
                s = [s]

            sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
            sys.stdout.flush()

            #New setparameters
            tmp_parameters = parameters.copy()

            j = 0
            for parameter in parameter_space:
                tmp_parameters[parameter] = s[j]
                j += 1

            self.model.saveParameters(tmp_parameters)
            self.model.run()

            tmp_V = open("tmp_V.p", "r")
            tmp_t = open("tmp_t.p", "r")

            #Get the results from the neuron run
            V = cPickle.load(tmp_V)
            t = cPickle.load(tmp_t)

            tmp_V.close()
            tmp_t.close()

            # Do a feature selection here. Make it so several feature
            # selections are performed at this step. Do this when
            # rewriting it as a class

            if feature != None:
                V = feature(V)

            if cvode_active:
                inter = scipy.interpolate.InterpolatedUnivariateSpline(t, V, k=3)
                self.solves.append((t, V, inter))
            else:
                self.solves.append((t, V))

            i += 1

        print "\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100)

        solves = np.array(solves)
        if cvode_active:
            lengths = []
            for s in self.solves[:,0]:
                lengths.append(len(s))          

            index_max_len = np.argmax(lengths)
            self.t_max = self.solves[index_max_len, 0]

            interpolated_solves = []
            for inter in self.solves[:,2]:
                interpolated_solves.append(inter(t_max))

        else:
            self.t_max = solves[0, 0]
            interpolated_solves = solves[:,1]

        self.U_hat = cp.fit_regression(P, nodes, interpolated_solves, rule="LS")
        






        

    def plotV_t(parameter):
        color1 = 0
        color2 = 8

        prettyPlot(self.t, self.E,
                   "Mean, " + parameter, "time", "voltage", color1)
        plt.savefig(os.path.join(self.outputdir, parameter  + "_mean" + self.figureformat),
                    bbox_inches="tight")

        prettyPlot(self.t, self.Var,
                   "Variance, " + parameter, "time", "voltage", color2)
        plt.savefig(os.path.join(self.outputdir, parameter  + "_variance" + self.figureformat),
                    bbox_inches="tight")


        ax, tableau20 = prettyPlot(self.t, self.E,
                                   "Mean and variance, " + parameter, "time", "voltage, mean", color1)
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off",  
                        labelright="on", color=tableau20[color2], labelcolor=tableau20[color2],
                        labelsize=14)
        ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
        ax.spines["right"].set_edgecolor(tableau20[color2])


        ax2.set_xlim([min(self.t),max(self.t)])
        ax2.set_ylim([min(self.Var),max(self.Var)])

        ax2.plot(self.t, self.Var, color=tableau20[color2], linewidth=2, antialiased=True)

        ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
        ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
        ax.spines["left"].set_edgecolor(tableau20[color1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputdir, parameter  + "_variance_mean" + self.figureformat), bbox_inches="tight")

        plt.close()



    def plotConfidenceInterval(filename):

        ax, color = prettyPlot(self.t_max, self.E, "Confidence interval", "time", "voltage", 0)
        plt.fill_between(self.t_max, self.p_10, self.p_90, alpha=0.2, facecolor = color[8])
        prettyPlot(self.t_max, self.p_90, color = 8, new_figure = False)
        prettyPlot(self.t_max, self.p_10, color = 9, new_figure = False)
        prettyPlot(self.t_max, self.E, "Confidence interval", "time", "voltage", 0, False)

        plt.ylim([min([min(self.p_90), min(self.p_10), min(self.E)]),
                  max([max(self.p_90), max(self.p_10), max(self.E)])])

        plt.legend(["Mean", "$P_{90}$", "$P_{10}$"])
        plt.savefig(os.path.join(self.outputdir, filename + self.figureformat),
                    bbox_inches="tight")

        plt.close()


    def plotSensitivity(self):

        for i in range(len(self.sensitivity)):
            prettyPlot(self.t_max, self.sensitivity[i], self.parameter_space.keys()[i] + " sensitivity", "time", "sensitivity", i, True)
            plt.title(self.parameter_space.keys()[i] + " sensitivity",)
            plt.ylim([0, 1.05])
            plt.savefig(os.path.join(self.outputdir, self.parameter_space.keys()[i] +"_sensitivity" + self.figureformat),
                    bbox_inches="tight")
        plt.close()

        for i in range(len(self.sensitivity)):
            prettyPlot(self.t_max, self.sensitivity[i], "sensitivity", "time", "sensitivity", i, False)

        plt.ylim([0, 1.05])
        plt.xlim([self.t_max[0], 1.3*self.t_max[-1]])
        plt.legend(self.parameter_space.keys())
        plt.savefig(os.path.join(self.outputdir, "all_sensitivity" + self.figureformat),
                    bbox_inches="tight")



    def singleParameters(fitted_parameters = fitted_parameters,
                         distribution = uniform, outputdir = figurepath):
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)

        for fitted_parameter in fitted_parameters:
            print "\rRunning for " + fitted_parameter + "                     "

            parameter_space =  newParameterSpace(fitted_parameter, distribution)
            tmp_results = createPCExpansion(parameter_space, cvode_active=True)
            if tmp_results == None:
                print "Calculations aborted for " + fitted_parameter
                continue
            try:    
                U_hat, dist, solves, t_max, P, nodes = tmp_results

                E = cp.E(U_hat, dist)
                Var =  cp.Var(U_hat, dist)

                plotV_t(t_max, E, Var, fitted_parameter, outputdir)

                samples = dist.sample(nr_mc_samples, "M")
                U_mc = U_hat(samples)

                p_10 = np.percentile(U_mc, 10, 1)
                p_90 = np.percentile(U_mc, 90, 1)

                plotConfidenceInterval(t_max, E, p_10, p_90, fitted_parameter + "_confidence_interval", outputdir)

            except MemoryError:
                print "Memory error, calculations aborted"
                return 1

        return 0

    def exploreSingleParameters(distribution_functions, intervals, outputdir = figurepath):
        for distribution_function in distribution_functions:
            print "Running for distribution: " + distribution_function.__name__.split("_")[0]

            for interval in intervals[distribution_function.__name__.lower().split("_")[0]]:
                folder_name =  distribution_function.__name__.lower().split("_")[0] + "_" + str(interval)
                current_outputdir = os.path.join(outputdir, folder_name)

                print "Running for interval: %2.4g" % (interval) 
                singleParameters(distribution = Distribution(distribution_function, interval), outputdir = current_outputdir)




    def allParameters(fitted_parameters = fitted_parameters, distribution = uniform, outputdir = figurepath):
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)


        parameter_space =  newParameterSpace(fitted_parameters, distribution)
        tmp_results = createPCExpansion(parameter_space, cvode_active=True)
        if tmp_results == None:
            print "Calculations aborted for " #Add which distribution when rewritten as class
            return 1
        U_hat, dist, solves, t_max, P, nodes  = tmp_results

        try:    
            E = cp.E(U_hat, dist)
            Var =  cp.Var(U_hat, dist)

            plotV_t(t_max, E, Var, "all", outputdir)

            sensitivity = cp.Sens_t(U_hat, dist)

            plotSensitivity(t_max, sensitivity, parameter_space, outputdir)

            samples = dist.sample(nr_mc_samples)

            U_mc = U_hat(*samples)
            p_10 = np.percentile(U_mc, 10, 1)
            p_90 = np.percentile(U_mc, 90, 1)


            plotConfidenceInterval(t_max, E, p_10, p_90, "all_confidence_interval", outputdir)

        except MemoryError:
            print "Memory error, calculations aborted"
            return 1

        return 0

    def exploreAllParameters(distribution_functions, intervals, outputdir = figurepath):
        for distribution_function in distribution_functions:
            print "Running for distribution: " + distribution_function.__name__.split("_")[0]

            for interval in intervals[distribution_function.__name__.lower().split("_")[0]]:
                folder_name =  distribution_function.__name__.lower().split("_")[0] + "_" + str(interval)
                current_outputdir = os.path.join(outputdir, folder_name)

                print "Running for interval: %2.4g" % (interval) 
                allParameters(distribution = Distribution(distribution_function, interval), outputdir = current_outputdir)



        
        

class Model():
    def __init__(self, modelfile, modelpath, parameterfile, filedir, memory_report = None):
        self.modelfile = modelfile
        self.modelpath = modelpath
        self.parameterfile = parameterfile
        self.filedir = filedir
        self.memory_report = memory_report

    def saveParameters(parameters):
        parameter_string = """
rall =    $rall 
cap =     $cap
Rm =      $Rm
Vrest =   $Vrest
Epas =    $Epas
gna =     $gna
nash =    $nash
gkdr =    $gkdr
kdrsh =   $kdrsh
gahp =    $gahp
gcat =    $gcat
gcal =    $gcal
ghbar =   $ghbar
catau =   $catau
gcanbar = $gcanbar
        """

        parameter_template = string.Template(parameter_string)
        filled_parameter_string = parameter_template.substitute(parameters)

        if os.path.samefile(os.getcwd(), os.path.join(self.filedir, self.modelpath)):
            f = open(self.parameterfile, "w")
        else:
            f = open(self.modelpath + self.parameterfile, "w")
        f.write(filled_parameter_string)
        f.close()

    def run():
        vdisplay = Xvfb()
        vdisplay.start()

        cmd = ["python", "simulation.py", self.parameterfile, self.modelfile, self.modelpath, str(cvode_active)]
        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #Note this checks total memory used by all applications
        if self.memory_report:
            while simulation.poll() == None:
                memory_report.save()
                memory_report.saveAll()
                if memory_report.totalPercent() > memory_threshold:
                    print "\nWARNING: memory threshold exceeded, aborting simulation"
                    simulation.terminate()
                    vdisplay.stop()
                    return None

                time.sleep(delta_poll)
        

        ut, err = simulation.communicate()

        vdisplay.stop()
        
        if simulation.returncode != 0:
            print "Error when running simulation:"
            print err
            sys.exit(1)

            
class Distribution():
    def __init__(self, function, interval):
        self.interval = interval
        self.function = function

    def __init__(self, interval):
        self.interval = interval
        
    def __call__(self, parameter):
        return self.function(parameter, self.interval)

        
    def normal(self, parameter):
        return cp.Normal(parameter, abs(self.interval*parameter))

    
    def uniform(self, parameter):
        return cp.Uniform(parameter - abs(self.interval*parameter),
                              parameter + abs(self.interval*parameter))


test_parameters =   ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat"]

test_parameters =   ["Rm", "Epas"]


singleParameters(distribution = Distribution(normal_function, 0.01), outputdir = figurepath + "test_single")
#allParameters(fitted_parameters = test_parameters, outputdir = figurepath + "test_all")
#
#allParameters(distribution = Distribution(normal_function, 0.1),
#              fitted_parameters = fitted_parameters, outputdir = figurepath + "test_all")
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



t_end = time.time()

subprocess.Popen(["play","-q","ship_bell.wav"])

print "The total runtime is: " + str(datetime.timedelta(seconds=(t_end-t_start)))
