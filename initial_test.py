import os
import numpy as np

os.chdir("./neuron_models/dLGN_modelDB/")

from neuron import h
h("forall delete_section()")
h.load_file("INmodel.hoc")
os.chdir("../../")

def toArray(hocObject):
    array = np.zeros(hocObject.size())
    hocObject.to_python(array)
    return array


    
# Global parameters
celsius = 36.0 # temperature
Epas = -70.6 # reversal potential for leakage current

# Parameters
rall = 513
cap = 1.1
Rm = 22000
Vrest = -63
#Epas = -67

# Channel densities & shifts (for additional channels see May_dends.hoc)
gna =  5.90
nash = - 52.6
gkdr = 0.37
kdrsh = -51.2
gahp = 6.4e-5
gcat=1.17e-5
gcal=0.000009
ghbar=0.0000011
catau = 50
gcanbar = 2e-8



def initialize():
    """
    Initialize the code
    """
    cvode = h.CVode()
    cvode.active(1)
    
    h.celsius = celsius
    print h.soma.Ra
    #h.soma.Ra = rall
    #print h.soma.Ra
    # for sec in h.soma:
    #     h.distance()

    #for sec in h.allsec():
        #sec.Ra = rall
        #print sec
    #     
    #     sec.e_pas = 30000
        #sec.ihdendfac = 34
    #     sec.v = Epas
    #     sec.e_pas = Epas

    #     sec.insert("pas")
    #     sec.e_pas = Epas
    #     sec.g_pas = 1/Rm
    #     sec.Ra = rall
    #     sec.cm = cap
    #     sec.gnabar_hh2 = 0
    #     sec.gkbar_hh2 = 0
    #     sec.gcabar_it2 = gcat
        
    # for sec in h.soma:
    #     sec.gnabar_hh2 = gna
    #     sec.gkbar_hh2 = gkdr
    #     sec.gcabar_it2 = gcat

    

    # stim = h.IClamp(.5)
    # stim.delay = 1000
    # stim.dur = 100
    # stim.amp = 0 #nA
    h("{Ra = %d}" % (511))
    h.finitialize()
    h.fcurrent()
    cvode.re_init()
    #print h.soma.Ra
    
    
initialize()
print h.soma.Ra

    



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

# run the simulation
#vec['t'].record(h._ref_t)
# vec['current'].record(VC_patch._ref_i)
#vec['stim_curr'].record(stim._ref_i)
h.load_file("stdrun.hoc")
h.tstop = 2500	# Simulation time
#h.t = -500

def record(ref_data):
    data = h.Vector()
    data.record(getattr(h, ref_data))
    return data

t = record("_ref_t")

h.run()
#time = np.zeros()
t = toArray(t)
#t = np.zeros(vec['t'].size())
#vec['t'].to_python(t)
#t = toArray(vec['t'])
V_sec = toArray(vec['V_sec'][0])
print vec['V_sec'][0]
print V_sec


import	matplotlib.pyplot as plt
# ########################################################################
## Plotting propagation of voltage signal
fig = plt.figure()

plt.plot(t, V_sec)
plt.show()
# ax1 = fig.add_subplot(4, 1, 1)
# ax1.set_title('tittel')
# ax1.text(-0.025, 1.025, 'A',horizontalalignment='center',verticalalignment='bottom',fontsize=18,
# 	 transform=ax1.transAxes)

# ax2 = fig.add_subplot(4, 1, 2, sharex=ax1, sharey=ax1, ylabel='Voltage(mV)')
# ax2.text(-0.025, 1.025, 'B',horizontalalignment='center',verticalalignment='bottom',fontsize=18,
# 	 transform=ax2.transAxes)

# ax3 = fig.add_subplot(4, 1, 3, sharex=ax1, sharey=ax1)
# ax3.text(-0.025, 1.025, 'C',horizontalalignment='center',verticalalignment='bottom',fontsize=18,
# 	 transform=ax3.transAxes)

# ax4 = fig.add_subplot(4, 1, 4, sharex=ax1, sharey=ax1)
# ax4.text(-0.025, 1.025, 'D',horizontalalignment='center',verticalalignment='bottom',fontsize=18,
# 	 transform=ax4.transAxes)


# # 0, 83, 88, 99 are selected points along a single dendritic branch
# ax1.plot(t, V_sec, label = 'At soma')
# #ax2.plot(vec['t'], vec['V_sec'][83], label = '%1.1f $\mu$m' %vec['d_sec'][83])
# #ax3.plot(vec['t'], vec['V_sec'][88], label = '%1.1f $\mu$m' % vec['d_sec'][88])
# #ax4.plot(vec['t'], vec['V_sec'][99], label = '%1.1f $\mu$m' % vec['d_sec'][99])
# ax1.legend(loc='best',frameon=False, fontsize=13)
# #ax2.legend(loc='best',frameon=False, fontsize=13)
# #ax3.legend(loc='best',frameon=False, fontsize=13)
# #ax4.legend(loc='best',frameon=False, fontsize=13)
# plt.setp([a.set_xlabel('') for a in [ax1,ax2,ax3]], visible=False)
# plt.setp([a.get_xticklabels() for a in [ax1,ax2,ax3]], visible=False)
# plt.show()
