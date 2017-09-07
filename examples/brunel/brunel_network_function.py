import nest
import nest.raster_plot
import matplotlib.pyplot as plt


def brunel_network(eta=2, g=5):
    """
    A brunel network, from:

    Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
    Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
    183-208 (2000).


    Parameters
    ----------
    g : {int, float}, optional
        Ratio inhibitory weight/excitatory weight. Default is 5.
    eta : {int, float}, optional
        External rate relative to threshold rate. Default is 2.
    Note that for testing purposes this network has fewer neurons (N_neurons = 100)
    and run for a shorter time (simtime = 40) and record from fewer neuron (N_rec = 10) than
    the brunel network class.
    """

    # Network parameters
    N_rec = 5              # Record from 50 neurons
    simtime = 50          # Simulation time

    # g = 5.0                # Ratio inhibitory weight/excitatory weight
    # eta = 2.0              # External rate relative to threshold rate
    delay = 1.5            # Synaptic delay in ms
    tau_m = 20.0           # Time constant of membrane potential in ms
    V_th = 20.0
    N_E = 8000             # Number of inhibitory neurons
    N_I = 2000             # Number of excitatory neurons
    N_neurons = N_E + N_I  # Number of neurons in total
    C_E = N_E/10           # Number of excitatory synapses per neuron
    C_I = N_I/10           # Number of inhibitory synapses per neuron
    J_E = 0.1              # Amplitude of excitatory postsynaptic current
    J_I = -g*J_E           # Amplitude of inhibitory postsynaptic current

    nu_ex = eta*V_th/(J_E*C_E*tau_m)
    p_rate = 1000.0*nu_ex*C_E

    nest.ResetKernel()

    # Configure kernel
    nest.SetKernelStatus({'print_time': False})


    nest.SetDefaults('iaf_psc_delta',
                     {'C_m': 1.0,
                      'tau_m': tau_m,
                      't_ref': 2.0,
                      'E_L': 0.0,
                      'V_th': V_th,
                      'V_reset': 10.0})


    # Create neurons
    nodes   = nest.Create('iaf_psc_delta', N_neurons)
    nodes_E = nodes[:N_E]
    nodes_I = nodes[N_E:]

    noise = nest.Create('poisson_generator',1,{'rate': p_rate})

    spikes = nest.Create('spike_detector',2,
                         [{'label': 'brunel-py-ex'},
                          {'label': 'brunel-py-in'}])
    spikes_E = spikes[:1]
    spikes_I = spikes[1:]


    # Connect neurons to each other
    nest.CopyModel('static_synapse_hom_w', 'excitatory',
                   {'weight':J_E, 'delay':delay})
    nest.Connect(nodes_E, nodes,
                 {'rule': 'fixed_indegree', 'indegree': C_E},
                 'excitatory')

    nest.CopyModel('static_synapse_hom_w', 'inhibitory',
                   {'weight': J_I, 'delay': delay})
    nest.Connect(nodes_I, nodes,
                {'rule': 'fixed_indegree', 'indegree': C_I},
                 'inhibitory')



    # Connect poisson generator to all nodes
    nest.Connect(noise, nodes, syn_spec='excitatory')

    nest.Connect(nodes_E[:N_rec], spikes_E)
    nest.Connect(nodes_I[:N_rec], spikes_I)


    # Run the simulation
    nest.Simulate(simtime)


    events_E = nest.GetStatus(spikes_E, 'events')[0]
    events_I = nest.GetStatus(spikes_I, 'events')[0]


    # TODO is there any difference in sending back only excitatory
    #      or inhibitory neurons, or should both be sent back?
    spiketrains = []

    # Excitatory spike trains
    for sender in set(events_E["senders"]):
        spiketrain = events_E["times"][events_E["senders"] == sender]
        spiketrains.append(spiketrain)

    # Inhibitory spike trains
    # for sender in set(events_I["senders"]):
    #     spiketrain = events_I["times"][events_I["senders"] == sender]
    #     spiketrains.append(spiketrain)

    # U must be a list/array of spiketrains
    U = spiketrains
    t = simtime

    return t, U




if __name__ == "__main__":
    t, U = brunel_network()
    print U