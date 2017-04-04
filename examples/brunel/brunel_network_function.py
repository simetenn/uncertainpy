import pandas as pd
import numpy as np
import nest
import nest.raster_plot

import matplotlib.pyplot as plt


def calc_CV(spikes):
    isi = np.diff(spikes)
    if len(isi) > 0:
        CV = np.sqrt(np.mean(isi**2) - np.mean(isi)**2) / np.mean(isi)
        return CV
    else:
        return np.nan



def brunel_network(J_E=0.5, g=5.0):
    # g = 5.0           # Ratio of IPSP to EPSP amplitude: J_I/J_E
    # J_E = 0.5         # J_I = -g*J_E


    # Network parameters. These are given in Brunel (2000) J.Comp.Neuro.
    eta = 1.0         # rate of external population in multiples of threshold rate
    delay = 1.5       # synaptic delay in ms
    C_m = 281.0
    tau_m = C_m/20.0  # Membrane time constant in mV
    V_th = -50.4      # Spike threshold in mV
    t_ref = 2.0
    E_L = -60.
    V_reset = -60.

    N_neurons = 1000
    P_E = .8           # P_I = 1 - P_E

    N_rec = 100        # Number of neurons to record from
    simtime = 100


    nest.ResetKernel()

    N_E = int(N_neurons*P_E)
    N_I = int(N_neurons*(1 - P_E))

    C_E = int(N_E/10)  # number of excitatory synapses per neuron
    C_I = int(N_I/10)  # number of inhibitory synapses per neuron

    J_I = -g*J_E
    # rate of an external neuron in ms^-1
    nu_ex = eta*abs(V_th)/(J_E*C_E*tau_m)
    # rate of the external population in s^-1
    p_rate = 1000.0*nu_ex*C_E

    # Configure kernel and neuron defaults
    nest.SetKernelStatus({"print_time": True,
                          "local_num_threads": 1})

    nest.SetDefaults("iaf_psc_delta",
                     {"C_m": C_m,
                      "tau_m": tau_m,
                      "t_ref": t_ref,
                      "E_L": E_L,
                      "V_th": V_th,
                      "V_reset": V_reset})
    # Create neurons
    nodes = nest.Create("iaf_psc_delta", N_neurons)
    nodes_E = nodes[:N_E]
    nodes_I = nodes[N_E:]

    # Connect neurons with each other
    nest.CopyModel("static_synapse", "excitatory",
                   {"weight": float(J_E), "delay": float(delay)})

    nest.Connect(nodes_E, nodes,
                 {"rule": 'fixed_indegree', "indegree": C_E},
                 "excitatory")

    nest.CopyModel("static_synapse", "inhibitory",
                   {"weight": float(J_I), "delay": float(delay)})

    nest.Connect(nodes_I, nodes,
                 {"rule": 'fixed_indegree', "indegree": C_I},
                 "inhibitory")

    # Add stimulation and recording devices
    noise = nest.Create("poisson_generator", params={"rate": p_rate})
    # connect using all_to_all: one noise generator to all neurons

    nest.Connect(noise, nodes, syn_spec="excitatory")

    spikes = nest.Create("spike_detector", 2,
                         params=[{"label": "Exc", "to_file": False},
                                 {"label": "Inh", "to_file": False}])
    spike_detec_E = spikes[:1]
    spike_detec_I = spikes[1:]

    # connect using all_to_all: all recorded excitatory neurons to one
    # detector
    nest.Connect(nodes_E[:N_rec], spike_detec_E)
    nest.Connect(nodes_I[:N_rec], spike_detec_I)


    # Run the simulation
    nest.Simulate(simtime)
    events_E = pd.DataFrame(nest.GetStatus(spike_detec_E, 'events')[0])

    cv = []
    for idx, sender in enumerate(events_E.senders):
        spikes = events_E[events_E.senders == sender].times
        cv.append(calc_CV(spikes))

    U = np.nanmean(np.array(cv))
    t = None

    return t, U
