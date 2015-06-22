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
figurepath = "figures/"

#fitted_parameters = ['Rm', 'Epas', 'ghbar', 'gcat']
fitted_parameters = ['Rm', 'Epas']


#Quick and dirty parameter space thingys
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


    
def setParameters(parameters):
    for param in parameters:
        setattr(h, param, parameters[param])
        
    h.finitialize()

def setParameters(parameters, parameter_values):
    if len(parameters) != len(parameter_values):
        print "Error: Length of parameters and parameter_values is not equal"
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

#h("cvode_active(0)")
def createPCExpansion(parameter_space = parameters, cvode_active=True):

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
 
        setParameters(fitted_parameters, s)
        h.run()
        
        V_ = toArray(V)
        t_ = toArray(t)

        if cvode_active:
            inter = scipy.interpolate.InterpolatedUnivariateSpline(t_, V_, k=3)
            #inter = scipy.interpolate.interp1d(t_, V_, kind="slinear")
            #inter = scipy.interpolate.BarycentricInterpolator(t_, V_)
            solves.append((t_, V_, inter))
        else:
            solves.append((t_, V_))

            
        
            
        i += 1
        
    sys.stdout.write("\rRunning Neuron: %2.1f%%" % (i/len(nodes.T)*100))
    sys.stdout.flush()    
    print ""
    
    solves = np.array(solves)
    if cvode_active:
        # Fix so we have the same amount of timesteps for each simulation.
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



    
#run(parameters)

#subprocess.call(["rm figures/tmp_image_*"], shell=True)

parameter_space =  newParameterSpaceUniform(fitted_parameters)
U_hat, dist, solves, t_max, P, nodes  = createPCExpansion(parameter_space, cvode_active=True)
U_hat_const, dist_const, solves_const, t_max_const, P_const, nodes_const  = createPCExpansion(parameter_space, cvode_active=False)


interpolated_solves = []
for inter in solves[:,2]:
    interpolated_solves.append(inter(t_max_const))

U_hat_const_like = cp.fit_regression(P, nodes, interpolated_solves, rule="LS")

plt.clf()
plt.plot(t_max, cp.E(U_hat, dist))
plt.title("Interpolation mean")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_interpolation_mean" + figureformat)

plt.clf()
plt.plot(t_max_const, cp.E(U_hat_const, dist_const))
plt.title("Constant dt mean")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_constant_dt_mean" + figureformat)


plt.clf()
plt.plot(t_max, cp.Var(U_hat, dist))
plt.title("Interpolation variance")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test_interpolation_variance" + figureformat)


plt.clf()
plt.plot(t_max_const, cp.Var(U_hat_const, dist_const))
plt.title("Constant dt variance")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_constant_dt_variance" + figureformat)


plt.clf()
plt.plot(t_max, cp.E(U_hat, dist))
plt.plot(t_max_const, cp.E(U_hat_const, dist_const))
plt.title("Mean constant dt vs interpolation")
plt.legend(["Interpolation", "Constant dt"])
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_comparison_mean" + figureformat)


plt.clf()
plt.plot(t_max, cp.Var(U_hat, dist))
plt.plot(t_max_const, cp.Var(U_hat_const, dist_const))
plt.title("Variance constant dt vs interpolation")
plt.legend(["Interpolation", "Constant dt"])
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_comparison_variance" + figureformat)

plt.clf()
plt.plot(t_max_const, abs(cp.E(U_hat_const, dist_const) - cp.E(U_hat_const_like, dist)))
plt.title("Difference in mean, constant dt vs interpolation")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_difference_mean" + figureformat)

plt.clf()
plt.plot(t_max_const, abs(cp.Var(U_hat_const, dist_const) - cp.Var(U_hat_const_like, dist)))
plt.title("Difference in variance, constant dt vs interpolation")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_difference_variance" + figureformat)

plt.clf()
plt.plot(t_max, cp.E(U_hat, dist))
plt.plot(t_max_const, cp.E(U_hat_const_like, dist_const))
plt.title("Mean constant dt vs interpolation")
plt.legend(["Interpolation", "Constant dt"])
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_comparison_mean_same_points" + figureformat)

plt.clf()
plt.plot(t_max, cp.Var(U_hat, dist))
plt.plot(t_max_const, cp.Var(U_hat_const_like, dist_const))
plt.title("Variance constant dt vs interpolation")
plt.legend(["Interpolation", "Constant dt"])
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.savefig(figurepath + "interpolation_test2_comparison_variance_same_points" + figureformat)


t_end = time.time()

subprocess.Popen(["play","-q","ship_bell.wav"])

#print "Interpolation time is: %g" % (t_constant_dt_end - t_interpolation_end)
#print "Interpolation time is: %g" % (t_interpolation_end - t_interpolation_start)
print "The total runtime is: " + str(datetime.timedelta(seconds=(t_end-t_start)))
