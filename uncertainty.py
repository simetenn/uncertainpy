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

t_start = time.time()

memory_report = Memory()

# Global parameters
interval = 5*10**-4
nr_mc_samples = 10**3

modelfile = "INmodel.hoc"
modelpath = "neuron_models/dLGN_modelDB/"
parameterfile = "Parameters.hoc" 

filepath = os.path.abspath(__file__)
filedir = os.path.dirname(filepath)
memory_threshold = 90
delta_poll = 1

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

figureformat = ".png"
figurepath = "figures/"
fitted_parameters =   ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat",
                        "gcal", "ghbar", "catau", "gcanbar"]




"""
 #Quick and dirty parameter space things
def newParameterSpaceNormal(fitted_parameters, parameters=parameters):
    parameter_space = {}
    for param in fitted_parameters:
        parameter_space[param] = cp.Normal(parameters[param], abs(std_percentage*parameters[param]))
    return parameter_space
        

def newParameterSpaceUniform(fitted_parameters, parameters=parameters):
    parameter_space = {}
    for param in fitted_parameters:
        parameter_space[param] = cp.Uniform(parameters[param] - abs(uniform_interval*parameters[param]),
                                             parameters[param] + abs(uniform_interval*parameters[param]))
    return parameter_space

"""


def plotV_t(t, E, Var, parameter, outputdir, figureformat = figureformat):
    color1 = 0
    color2 = 8

    prettyPlot(t, E,
               "Mean, " + parameter, "time", "voltage", color1)
    plt.savefig(os.path.join(outputdir, parameter  + "_mean" + figureformat),
                bbox_inches="tight")

    prettyPlot(t, Var,
               "Variance, " + parameter, "time", "voltage", color2)
    plt.savefig(os.path.join(outputdir, parameter  + "_variance" + figureformat),
                bbox_inches="tight")


    ax, tableau20 = prettyPlot(t, E,
                               "Mean and variance, " + parameter, "time", "voltage, mean", color1)
    ax2 = ax.twinx()
    ax2.tick_params(axis="y", which="both", right="on", left="off",  
                    labelright="on", color=tableau20[color2], labelcolor=tableau20[color2],
                    labelsize=14)
    ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
    ax.spines["right"].set_edgecolor(tableau20[color2])


    ax2.set_xlim([min(t),max(t)])
    ax2.set_ylim([min(Var),max(Var)])

    ax2.plot(t, Var, color=tableau20[color2], linewidth=2, antialiased=True)

    ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
    ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
    ax.spines["left"].set_edgecolor(tableau20[color1])
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, parameter  + "_variance_mean" + figureformat), bbox_inches="tight")

    plt.close()



def plotConfidenceInterval(t_max, E, p_10, p_90, filename, outputdir = figurepath):
  
    ax, color = prettyPlot(t_max, E, "Confidence interval", "time", "voltage", 0)
    plt.fill_between(t_max, p_10, p_90, alpha=0.2, facecolor = color[8])
    prettyPlot(t_max, p_90, color = 8, new_figure = False)
    prettyPlot(t_max, p_10, color = 9, new_figure = False)
    prettyPlot(t_max, E, "Confidence interval", "time", "voltage", 0, False)

    plt.ylim([min([min(p_90), min(p_10), min(E)]), max([max(p_90), max(p_10), max(E)])])
    
    plt.legend(["Mean", "$P_{90}$", "$P_{10}$"])
    plt.savefig(os.path.join(outputdir, filename + figureformat),
                bbox_inches="tight")

    plt.close()


def plotSensitivity(t_max, sensitivity, parameter_space, outputdir = figurepath):

    for i in range(len(sensitivity)):
        prettyPlot(t_max, sensitivity[i], parameter_space.keys()[i] + " sensitivity", "time", "sensitivity", i, True)
        plt.title(parameter_space.keys()[i] + " sensitivity",)
        plt.ylim([0, 1.05])
        plt.savefig(os.path.join(outputdir, parameter_space.keys()[i] +"_sensitivity" + figureformat),
                bbox_inches="tight")
    plt.close()
   
    for i in range(len(sensitivity)):
        prettyPlot(t_max, sensitivity[i], "sensitivity", "time", "sensitivity", i, False)

    plt.ylim([0, 1.05])
    plt.xlim([t_max[0], 1.3*t_max[-1]])
    plt.legend(parameter_space.keys())
    plt.savefig(os.path.join(outputdir, "all_sensitivity" + figureformat),
                bbox_inches="tight")


    

def newParameterSpace(fitted_parameters, distribution):
    """
    Generalized parameter space creation
    """

    parameter_space = {}

    if type(fitted_parameters) == str:
        parameter_space[fitted_parameters] = distribution(parameters[fitted_parameters])
        return parameter_space
    else:
        for param in fitted_parameters:
            parameter_space[param] = distribution(parameters[param])

        return parameter_space
        

        
def saveParameters(parameters, parameterfile, modelpath, filedir):
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

    if os.path.samefile(os.getcwd(), os.path.join(filedir, modelpath)):
        f = open(parameterfile, "w")
    else:
        f = open(modelpath + parameterfile, "w")
    f.write(filled_parameter_string)
    f.close()


def createPCExpansion(parameter_space, feature = None, cvode_active=True):

    dist = cp.J(*parameter_space.values())
    P = cp.orth_ttr(M, dist)
    nodes = dist.sample(2*len(P), "M")
    solves = []

    vdisplay = Xvfb()
    vdisplay.start()

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
           
        saveParameters(tmp_parameters, parameterfile, modelpath, filedir)

        cmd = ["python", "simulation.py", parameterfile, modelfile, modelpath, str(cvode_active)]
        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #Note this checks total memory used by all applications
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
        
        if simulation.returncode != 0:
            print "Error when running simulation:"
            print err
            sys.exit(1)
        

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
            solves.append((t, V, inter))
        else:
            solves.append((t, V))

        i += 1
   
    print "\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100)
#    sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
#    sys.stdout.flush()
      
    solves = np.array(solves)
    if cvode_active:
        lengths = []
        for s in solves[:,0]:
            lengths.append(len(s))          
           
        index_max_len = np.argmax(lengths)
        t_max = solves[index_max_len, 0]
            
        interpolated_solves = []
        for inter in solves[:,2]:
            interpolated_solves.append(inter(t_max))
         
    else:
        t_max = solves[0, 0]
        interpolated_solves = solves[:,1]


    vdisplay.stop()
    
    U_hat = cp.fit_regression(P, nodes, interpolated_solves, rule="LS")
    return U_hat, dist, solves, t_max, P, nodes



    
class Distribution():
    def __init__(self, function, interval):
        self.interval = interval
        self.function = function
        
    def __call__(self, parameter):
        return self.function(parameter, self.interval)

    
def normal_function(parameter, interval):
    return cp.Normal(parameter, abs(interval*parameter))

    
def uniform_function(parameter, interval):
    return cp.Uniform(parameter - abs(interval*parameter),
                      parameter + abs(interval*parameter))


    
normal = Distribution(normal_function, interval)
uniform = Distribution(uniform_function, interval)
"""
def Normal(parameter):
    return cp.Normal(parameter, abs(interval*parameter))

    
def Uniform(parameter):
    return cp.Uniform(parameter - abs(interval*parameter),
                      parameter + abs(interval*parameter))
    
"""
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

            #print
           #print len(t_max)
            #print U_hat(*parameters.values()).shape
            #plt.figure(1009)
            #print U_hat(parameters.values())
            #print U_hat(*parameters.values())
            #print max(U_hat(*parameters.values()))
            print parameters[fitted_parameter]
            plt.plot(U_hat(parameters[fitted_parameter]))
            plt.show()
            
            E = cp.E(U_hat, dist)
            Var =  cp.Var(U_hat, dist)

            plotV_t(t_max, E, Var, fitted_parameter, outputdir)

            samples = dist.sample(nr_mc_samples, "M")
            U_mc = U_hat(samples)
            print samples[12]
            plt.plot(U_hat(samples[12]))
            plt.show()
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
