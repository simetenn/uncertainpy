import time, subprocess, datetime
t_start = time.time()

import os, sys
import numpy as np
import scipy as scp
import scipy.interpolate
os.chdir("./neuron_models/dLGN_modelDB/")

from neuron import h
h("forall delete_section()")
h.load_file("INmodel.hoc")
os.chdir("../../")

import matplotlib.pyplot as plt
import chaospy as cp

std_percentage = 0.01
uniform_interval = 0.1

def record(ref_data):
    data = h.Vector()
    data.record(getattr(h, ref_data))
    return data
    

def toArray(hocObject):
    array = np.zeros(hocObject.size())
    hocObject.to_python(array)
    return array

    

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
figurepath = "/figures/"
#fitted_parameters =   ["Rm", "Epas", "gkdr", "kdrsh", "gahp", "gcat",
#                       "gcal", "ghbar", "catau", "gcanbar"]
#fitted_parameters = "gkdr"
#fitted_parameters = ["Rm","gahp"]
#fitted_parameters =   ["kdrsh", "Epas"]
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

def setParameters(parameter_values, parameters = fitted_parameters):
    if parameter_values.size == 1:
        parameter_values = [parameter_values]
    
    if len(parameters) != len(parameter_values):
        print "Error: Length of parameters and parameter_values are not equal"
        sys.exit(1)

    for i in range(len(parameters)):
        setattr(h, parameters[i], parameter_values[i])
        
    h.finitialize()


def run(parameters):

    setParameters(parameters)
    h.run()

    _t = toArray(t)
    _V = toArray(V)
    return _t,_V

    

def createPCExpansion(parameter_space, cvode_active=True):
        
    if not cvode_active:
        h("cvode_active(0)")
    
    for sec in h.allsec():
        V = h.Vector()
        V.record(sec(0.5)._ref_v)
        break

    t = record("_ref_t")
    
    dist = cp.J(*parameter_space.values())
    P = cp.orth_ttr(2, dist)
    nodes = dist.sample(2*len(P), "M")
    solves = []

    i = 0.
    for s in nodes.T:
        sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
        sys.stdout.flush()
        
        setParameters(s, parameter_space.keys())
        h.run()
        
        V_ = toArray(V)
        t_ = toArray(t)

        if cvode_active:
            inter = scipy.interpolate.InterpolatedUnivariateSpline(t_, V_, k=3)
            solves.append((t_, V_, inter))
        else:
            solves.append((t_, V_))

        i += 1
        
    sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
    sys.stdout.flush()    
    print ""
    
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


#subprocess.call(["rm figures/tmp_image_*"], shell=True)

#parameter_space =  newParameterSpaceNormal(fitted_parameters)

def Normal(parameter):
    return cp.Normal(parameter, abs(std_percentage*parameter))

def Uniform(parameter):
    return cp.Uniform(parameter - abs(uniform_interval*parameter),
                      parameter + abs(uniform_interval*parameter))

"""
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

def singleParameters(parameters = parameters):
    for parameter in parameters:
        print parameter
    parameter_space =  newParameterSpace(fitted_parameters, Uniform)
    U_hat, dist, solves, t_max, P, nodes  = createPCExpansion(parameter_space, cvode_active=True)


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

    

t_end = time.time()

#subprocess.Popen(["play","-q","ship_bell.wav"])

#print "Interpolation time is: %g" % (t_constant_dt_end - t_interpolation_end)
#print "Interpolation time is: %g" % (t_interpolation_end - t_interpolation_start)
print "The total runtime is: " + str(datetime.timedelta(seconds=(t_end-t_start)))
