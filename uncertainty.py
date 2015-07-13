### TODO
# Test out different types of polynomial chaos


import time, subprocess, datetime
t_start = time.time()

import os, sys, string
import numpy as np
import scipy as scp
import scipy.interpolate

filepath = os.path.abspath(__file__)
filedir = os.path.dirname(filepath)
modelfile = "INmodel.hoc"
modelpath = "neuron_models/dLGN_modelDB/"
parameterfile = "Parameters.hoc" 

os.chdir(modelpath)
from neuron import h
#h("forall delete_section()")
h.load_file(1, modelfile)
os.chdir(filedir)

import chaospy as cp




# Global parameters
interval = 0.05


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
#fitted_parameters =   ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat",
#                       "gcal", "ghbar", "catau", "gcanbar"]
#fitted_parameters = "gkdr"
#fitted_parameters = ["Rm","gahp"]
#fitted_parameters =   ["kdrsh", "Epas"]
fitted_parameters =   ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat",
                       "gcal", "ghbar", "catau", "gcanbar"]


class Simulation():
    def __init__(self, parameters, modelfile = modelfile, modelpath = modelpath, cvode_active = True):
        self.parameters = parameters
        self.modelfile = modelfile
        self.modelpath = modelpath
        self.cvode_active = cvode_active
        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(self.filepath)

        os.chdir(self.modelpath)
        self.saveParameters()

        
        import neuron
        self.h = neuron.h

        self.h.load_file(self.modelfile)
        os.chdir(self.filedir)

        if cvode_active:
            ""
            #self.h("cvode_active(1)")
        else:
            self.h("cvode_active(0)")


            
            
    def saveParameters(self):
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
        filled_parameter_string = parameter_template.substitute(self.parameters)

        if os.path.samefile(os.getcwd(), os.path.join(self.filedir, self.modelpath)):
            f = open(parameterfile, "w")
        else:
            f = open(modelpath + parameterfile, "w")
        f.write(filled_parameter_string)
        f.close()


        ### Be really careful with these. Need to make sure that all references to neuron are isnide this class
    def record(self, ref_data):
        data = self.h.Vector()
        data.record(getattr(self.h, ref_data))
        return data
    

    def toArray(self, hocObject):
        array = np.zeros(hocObject.size())
        hocObject.to_python(array)
        return array


    def recordV(self):
        for sec in self.h.allsec():
            self.V = self.h.Vector()
            self.V.record(sec(0.5)._ref_v)
            break

            
    def recordT(self):
        self.t = self.record("_ref_t")


    def run(self):
        self.h.finitialize()
        self.h.run()

        
    def getT(self):
        return self.toArray(self.t)

        
    def getV(self):
        return self.toArray(self.V)


        
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



def newParameterSpace(fitted_parameters, distribution = lambda parameter: cp.Normal(parameter, abs(std_percentage*parameter)), parameters=parameters):
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

"""
def setParameters(parameters):
    for param in parameters:
        setattr(h, param, parameters[param])
        
    h.finitialize()
"""



def reloadModel():
    print "reloading"

    #h("forall delete_section()")
    h.quit()
    # h.topology()
    print "sections"
    os.chdir(modelpath)
    h.load_file(1, modelfile)
    os.chdir(filedir)

    #h.load_file(modelfile)


def setParameters(parameter_values, parameters = fitted_parameters):
    if parameter_values.size == 1:
        parameter_values = [parameter_values]
    
    if len(parameters) != len(parameter_values):
        print "Error: Length of parameters and parameter_values are not equal"
        sys.exit(1)
      
    for i in range(len(parameters)):
        setattr(h, parameters[i], parameter_values[i])

        
    h.finitialize()


#def changeParameters(parameters = parameters):
#    saveParameters(parameters)
#    reloadModel()
    
   
#def record(ref_data):
#    data = h.Vector()
#    data.record(getattr(h, ref_data))
#    return data
    

#def toArray(hocObject):
#    array = np.zeros(hocObject.size())
#    hocObject.to_python(array)
#    return array
 

#def run(parameters):

#    setParameters(parameters)
#    h.run()

#    _t = toArray(t)
#    _V = toArray(V)
#    return _t,_V



    

def createPCExpansion(parameter_space, cvode_active=True):
 
    dist = cp.J(*parameter_space.values())
    P = cp.orth_ttr(2, dist)
    nodes = dist.sample(2*len(P), "M")
    solves = []
    
    i = 0.
    for s in nodes.T:
        sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
        sys.stdout.flush()

        #New setparameters
        tmpParameters = parameters.copy()
        for parameter in parameter_space:
            tmpParameters[parameter] = s

        sim = Simulation(tmpParameters, cvode_active = cvode_active)

        sim.recordT()
        sim.recordV()

        sim.run()
        #h.run()
        
        V = sim.getV()
        t = sim.getT()

        #del sim
        sim = None
        
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

    U_hat = cp.fit_regression(P, nodes, interpolated_solves, rule="LS")
    return U_hat, dist, solves, t_max, P, nodes



    
import matplotlib.pyplot as plt
def prettyPlot(x, y, title, xlabel, ylabel, color):
    

    axis_grey = (0.5,0.5,0.5)
    titlesize = 18
    fontsize = 16
    labelsize = 14
    # These are the "Tableau 20" colors as RGB.  
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
  
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.)  

    if plt.gcf() == "None":
        plt.figure(figsize=(10, 7.5))
    else:
        plt.clf()
    ax = plt.subplot(111)

    #for spine in ax.spines:
    #    ax.spines[spine].set_edgecolor(axis_grey)

    ax.spines["top"].set_edgecolor("None") 
    ax.spines["bottom"].set_edgecolor(axis_grey)
    ax.spines["right"].set_edgecolor("None")
    ax.spines["left"].set_edgecolor(axis_grey)


    ax.tick_params(axis="x", which="both", bottom="on", top="off",  
                    labelbottom="on", color=axis_grey, labelcolor="black",
                    labelsize=labelsize)  
    ax.tick_params(axis="y", which="both", right="off", left="on",  
                    labelleft="on", color=axis_grey, labelcolor="black",
                    labelsize=labelsize)  
        
    ax.plot(x, y, color=tableau20[color], linewidth=2, antialiased=True)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_xlim([min(x),max(x)])
    ax.set_ylim([min(y),max(y)])

    
    return ax, tableau20

def Normal(parameter):
    return cp.Normal(parameter, abs(interval*parameter))

def Uniform(parameter):
    return cp.Uniform(parameter - abs(interval*parameter),
                      parameter + abs(interval*parameter))


def singleParameters(parameters = parameters, distribution = Uniform, outputdir = figurepath):
    for parameter in parameters:
        sys.stdout.write("\r                                 ")
        sys.stdout.write("\rRunning for " + parameter)
        sys.stdout.flush()
        print
               
        parameter_space =  newParameterSpace(parameter, distribution)
        U_hat, dist, solves, t_max, P, nodes  = createPCExpansion(parameter_space, cvode_active=True)

        color1 = 0
        color2 = 8
        E = cp.E(U_hat, dist)
        Var =  cp.Var(U_hat, dist)

        #Plot mean
        prettyPlot(t_max, E,
                   "Mean, " + parameter, "time", "voltage", color1)
        plt.savefig(os.path.join(outputdir, parameter  + "_mean" + figureformat),
                    bbox_inches="tight")
        
        prettyPlot(t_max, Var,
                   "Variance, " + parameter, "time", "voltage", color2)
        plt.savefig(os.path.join(outputdir, parameter  + "_variance" + figureformat),
                    bbox_inches="tight")
            

        ax, tableau20 = prettyPlot(t_max, E,
                                   "Mean and variance, " + parameter, "time", "voltage, mean", color1)
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", which="both", right="on", left="off",  
                        labelright="on", color=tableau20[color2], labelcolor=tableau20[color2],
                        labelsize=14)
        ax2.set_ylabel('voltage, variance', color=tableau20[color2], fontsize=16)
        ax.spines["right"].set_edgecolor(tableau20[color2])
        
        
        ax2.set_xlim([min(t_max),max(t_max)])
        ax2.set_ylim([min(Var),max(Var)])
        

        ax2.plot(t_max, Var, color=tableau20[color2], linewidth=2, antialiased=True)

        ax.tick_params(axis="y", color=tableau20[color1], labelcolor=tableau20[color1])
        ax.set_ylabel('voltage, mean', color=tableau20[color1], fontsize=16)
        ax.spines["left"].set_edgecolor(tableau20[color1])
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, parameter  + "_variance_mean" + figureformat), bbox_inches="tight")
        
        plt.close()

    print



def exploreSingleParameters(distributions, intervals, outputdir = figurepath):
    global interval
    for distribution in distributions:
        #current_outputdir = os.path.join(outputdir, distribution.__name__.lower())
        #if not os.path.isdir(current_outputdir):
        #    os.mkdir(current_outputdir)
        
        print "Running for distribution: " + distribution.__name__
        for inter in intervals[distribution.__name__]:
            folder_name =  distribution.__name__.lower() + "_" + str(inter)
            current_outputdir = os.path.join(outputdir, folder_name)

            if not os.path.isdir(current_outputdir):
                os.mkdir(current_outputdir)
                  
            interval = inter

            print "Running for interval: %2.4f" % (interval) 
            singleParameters(distribution = distribution, outputdir = current_outputdir)


singleParameters()
#n_intervals = 10
#interval_range = {"Normal": np.linspace(0.001, 0.1, n_intervals), "Uniform":np.linspace(0.0005, 0.05, n_intervals)} 
#distributions = [Uniform, Normal]
#exploreSingleParameters(distributions, interval_range)       


"""
singleParameters()   


parameter_space =  newParameterSpace(fitted_parameters, Uniform)
U_hat, dist, solves, t_max, P, nodes  = createPCExpansion(parameter_space, cvode_active=True)
#U_hat_const, dist_const, solves_const, t_max_const, P_const, nodes_const  = createPCExpansion(parameter_space, cvode_active=False)


plt.figure()
plt.plot(t_max, cp.E(U_hat, dist))
plt.title("Mean")
plt.xlabel("Time")
plt.ylabel("Voltage")
#plt.savefig(figurepath + "mean" + figureformat)


plt.figure()
plt.plot(t_max, cp.Var(U_hat, dist))
plt.title("Variance")
plt.xlabel("Time")
plt.ylabel("Voltage")
#plt.savefig(figurepath + "variance" + figureformat)

plt.show()
"""

t_end = time.time()

subprocess.Popen(["play","-q","ship_bell.wav"])

print "The total runtime is: " + str(datetime.timedelta(seconds=(t_end-t_start)))
