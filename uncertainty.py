### TODO
# Test out different types of polynomial chaos methods
# Add sensitivity analysis
# Do dependent variable stuff
# Do a mc analysis after u_hat is generated



import time, subprocess, datetime, cPickle, scipy.interpolate, os, sys, string
import numpy as np
import scipy as scp
import chaospy as cp  
import matplotlib.pyplot as plt
from prettyPlot import prettyPlot
from xvfbwrapper import Xvfb

t_start = time.time()

# Global parameters
interval = 5*10**-7

modelfile = "INmodel.hoc"
modelpath = "neuron_models/dLGN_modelDB/"
parameterfile = "Parameters.hoc" 

filepath = os.path.abspath(__file__)
filedir = os.path.dirname(filepath)



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


def createPCExpansion(parameter_space, cvode_active=True):

    dist = cp.J(*parameter_space.values())
    P = cp.orth_ttr(2, dist)
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
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pipe.communicate()

        if pipe.returncode != 0:
            print "Error when running Neuron:"
            print err
            sys.exit(1)
        

        tmp_V = open("tmp_V.p", "r")
        tmp_t = open("tmp_t.p", "r")
        
        V = cPickle.load(tmp_V)
        t = cPickle.load(tmp_t)

        tmp_V.close()
        tmp_t.close()
        
        if cvode_active:
            inter = scipy.interpolate.InterpolatedUnivariateSpline(t, V, k=3)
            solves.append((t, V, inter))
        else:
            solves.append((t, V))

        i += 1
        
    sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
    sys.stdout.flush()
      
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

def Normal(parameter):
    return cp.Normal(parameter, abs(interval*parameter))

    
def Uniform(parameter):
    return cp.Uniform(parameter - abs(interval*parameter),
                      parameter + abs(interval*parameter))
    

def singleParameters(parameters = parameters, distribution = uniform, outputdir = figurepath):
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    for parameter in parameters:
        sys.stdout.write("\r                                 ")
        sys.stdout.write("\rRunning for " + parameter)
        sys.stdout.flush()
        print
               
        parameter_space =  newParameterSpace(parameter, distribution)
        U_hat, dist, solves, t_max, P, nodes  = createPCExpansion(parameter_space, cvode_active=True)
        
        E = cp.E(U_hat, dist)
        Var =  cp.Var(U_hat, dist)

        plotV_t(t_max, E, Var, parameter, outputdir)
   
    print



def exploreSingleParameters(distribution_functions, intervals, outputdir = figurepath):
    for distribution_function in distribution_functions:
        print "Running for distribution: " + distribution_function.__name__.split("_")[0]

        for interval in intervals[distribution_function.__name__.lower().split("_")[0]]:
            folder_name =  distribution_function.__name__.lower().split("_")[0] + "_" + str(interval)
            current_outputdir = os.path.join(outputdir, folder_name)
            
            print "Running for interval: %2.4f" % (interval) 
            singleParameters(distribution = Distribution(distribution_function, interval), outputdir = current_outputdir)



            
def allParameters(distribution = uniform, outputdir = figurepath):
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    
    parameter_space =  newParameterSpace(parameters, distribution)
    
    U_hat, dist, solves, t_max, P, nodes  = createPCExpansion(parameter_space, cvode_active=True)
        
    E = cp.E(U_hat, dist)
    Var =  cp.Var(U_hat, dist)

    plotV_t(t_max, E, Var, "all", outputdir)
    


def exploreAllParameters(distribution_functions, intervals, outputdir = figurepath):
    for distribution_function in distribution_functions:
        print "Running for distribution: " + distribution_function.__name__.split("_")[0]

        for interval in intervals[distribution_function.__name__.lower().split("_")[0]]:
            folder_name =  distribution_function.__name__.lower().split("_")[0] + "_" + str(interval)
            current_outputdir = os.path.join(outputdir, folder_name)
            
            print "Running for interval: %2.4g" % (interval) 
            allParameters(distribution = Distribution(distribution_function, interval), outputdir = current_outputdir)


    
#singleParameters(outputdir = figurepath + "single/test")
#allParameters(outputdir = figurepath + "all/test")


n_intervals = 10
distributions = [uniform_function, normal_function]
interval_range = {"normal" : np.linspace(10**-4, 10**-1, n_intervals),
                  "uniform" : np.linspace(5*10**-5, 5*10**-1, n_intervals)} 
exploreSingleParameters(distributions, interval_range, figurepath + "single")


n_intervals = 10
interval_range = {"normal" : np.linspace(10**-7, 10**-5, n_intervals),
                  "uniform" : np.linspace(5*10**-8, 5*10**-6, n_intervals)}
exploreAllParameters(distributions, interval_range, figurepath + "all")





t_end = time.time()

subprocess.Popen(["play","-q","ship_bell.wav"])

print "The total runtime is: " + str(datetime.timedelta(seconds=(t_end-t_start)))
