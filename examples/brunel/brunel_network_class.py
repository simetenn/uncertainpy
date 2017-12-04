from uncertainpy import NestModel

import numpy as np
import nest
import nest.raster_plot

import matplotlib.pyplot as plt

class BrunelNetwork(NestModel):
    def __init__(self):
        super(BrunelNetwork, self).__init__()

        # Network parameters. These are given in Brunel (2000) J.Comp.Neuro.
        self.g = 5.0  # Ratio of IPSP to EPSP amplitude: J_I/J_E
        self.eta = 1.0  # rate of external population in multiples of threshold rate
        self.delay = 1.5  # synaptic delay in ms
        self.C_m = 281.0
        self.tau_m = self.C_m/20.0   # Membrane time constant in mV
        self.V_th = -50.4  # Spike threshold in mV
        self.t_ref = 2.0
        self.E_L = -60.
        self.V_reset = -60.

        self.J_E = 0.5  # J_I = -g*J_E

        self.N_neurons = 1000
        self.P_E = .8  # P_I = 1 - P_E

        self.N_rec = 100   # Number of neurons to record from
        self.simtime = 100




    def reset(self):
        nest.ResetKernel()

        N_E = int(self.N_neurons * self.P_E)
        N_I = int(self.N_neurons * (1 - self.P_E))

        C_E = int(N_E/10)  # number of excitatory synapses per neuron
        C_I = int(N_I/10)  # number of inhibitory synapses per neuron

        J_I = -self.g * self.J_E
        # rate of an external neuron in ms^-1
        nu_ex = self.eta * abs(self.V_th) / (self.J_E * C_E * self.tau_m)
        # rate of the external population in s^-1
        p_rate = 1000.0 * nu_ex * C_E

        # Configure kernel and neuron defaults
        nest.SetKernelStatus({"print_time": True,
                              "local_num_threads": 1})

        nest.SetDefaults("iaf_psc_delta",
                         {"C_m": self.C_m,
                          "tau_m": self.tau_m,
                          "t_ref": self.t_ref,
                          "E_L": self.E_L,
                          "V_th": self.V_th,
                          "V_reset": self.V_reset})
        # Create neurons
        nodes = nest.Create("iaf_psc_delta", self.N_neurons)
        nodes_E = nodes[:N_E]
        nodes_I = nodes[N_E:]

        # Connect neurons with each other
        nest.CopyModel("static_synapse", "excitatory",
                       {"weight": float(self.J_E), "delay": float(self.delay)})

        nest.Connect(nodes_E, nodes,
                     {"rule": 'fixed_indegree', "indegree": C_E},
                     "excitatory")

        nest.CopyModel("static_synapse", "inhibitory",
                       {"weight": float(J_I), "delay": float(self.delay)})

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
        self.spike_detect_E = spikes[:1]
        self.spike_detect_I = spikes[1:]

        # connect using all_to_all: all recorded excitatory neurons to one
        # detector
        nest.Connect(nodes_E[:self.N_rec], self.spike_detect_E)
        nest.Connect(nodes_I[:self.N_rec], self.spike_detect_I)


    def run(self, **parameters):
        self.set_parameters(**parameters)

        self.reset()

        nest.Simulate(self.simtime)
        events = nest.GetStatus(self.spike_detect_E, 'events')[0]

        spiketrains = []
        for sender in set(events["senders"]):
            spiketrain = events["times"][events["senders"] == sender]
            spiketrains.append(spiketrain)

        # values must be a list/array of spiketrains
        time = self.simtime

        return time, spiketrains