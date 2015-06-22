import os
import numpy as np
from change_parameters import changeParameters
#from operator import itemgetter

os.chdir("./neuron_models/dLGN_modelDB/")

from neuron import h
h("forall delete_section()")
h.load_file("INmodel.hoc")
os.chdir("../../")

import	matplotlib.pyplot as plt


def record(ref_data):
    data = h.Vector()
    data.record(getattr(h, ref_data))
    return data
    

def toArray(hocObject):
    array = np.zeros(hocObject.size())
    hocObject.to_python(array)
    return array

    

parameters = {
    "rall" : 113,   
    "cap" : 1.1,
    "Rm" : 22000,
    "Vrest" : -63,
    "Epas" : -67,
    "gna" :  0.09,
    "nash" : -52.6,
    "gkdr" : 0.37,
    "kdrsh" : -51.2,
    "gahp" : 6.4e-5,
    "gcat" :1.17e-5,
    "gcal" :0.0009,
    "ghbar" :0.00011,
    "catau" : 50,
    "gcanbar" : 2e-8
}


def setParameters(parameters):
    for param in parameters:
        setattr(h, param, parameters[param])
        
    h.finitialize()

vec ={}
for var in 't', 'd_sec', 'd_seg', 'diam_sec','gc','diam_seg','stim_curr':
    vec[var] = h.Vector()

for var in 'V_sec', 'V_seg', 'CaConc_sec','CaConc_seg':
    vec[var] = h.List()

def create_lists(vec):
    for sec in h.allsec():
	vec['d_sec'].append(h.distance(1))
	vec['diam_sec'].append(sec.diam)
	rec0 = h.Vector()
	rec0.record(sec(0.5)._ref_v)
	vec['V_sec'].append(rec0)
	rec_Ca = h.Vector()
	rec_Ca.record(sec(0.5)._ref_Cai)
	vec['CaConc_sec'].append(rec_Ca)
	for seg in sec:
	    vec['d_seg'].append(h.distance(0) + sec.L * seg.x)
	    vec['diam_seg'].append(seg.diam)
	    vec['gc'].append(seg.gcabar_it2)
	    rec = h.Vector()
	    rec.record(seg._ref_v)
	    vec['V_seg'].append(rec)
	    rec1 = h.Vector()
	    rec1.record(seg._ref_Cai)
	    vec['CaConc_seg'].append(rec1)
        return vec
	    
create_lists(vec)


#changeParameters(parameters)
#h.load_file("./neuron_models/dLGN_modelDB/Parameters.hoc")

def run(parameters):

    for param in parameters:
        setattr(h, param, parameters[param])
        
    h.finitialize()

    t = record("_ref_t")

    h.run()

    t = toArray(t)
    V_sec = toArray(vec['V_sec'][0])

    fig = plt.figure()
    
    plt.plot(t, V_sec)
    plt.show()


run(parameters)
parameters["rall"] = 313
run(parameters)
