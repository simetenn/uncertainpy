import nest

def brunel_network(eta, g, delay, J_E):
    """
    A brunel network, from:

    Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
    Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
    183-208 (2000).

    Implementation adapted from:
    http://www.nest-simulator.org/py_sample/random-balanced-network-exp-synapses-multiple-time-constants/

    Parameters
    ----------
    g : {int, float}, optional
        Ratio inhibitory weight/excitatory weight. Default is 5.
    eta : {int, float}, optional
        External rate relative to threshold rate. Default is 2.
    delay : {int, float}, optional
        Synaptic delay in ms. Default is 1.5.
    J_E : {int, float}, optional
        Amplitude of excitatory postsynaptic current. Default is 0.1
    """

    # Reduced number of neurons and simulation time
    # Network parameters
    N_rec = 20             # Record from 20 neurons
    simulation_end = 100   # Simulation time

    tau_m = 20.0           # Time constant of membrane potential in ms
    V_th = 20.0
    N_E = 1000             # Number of inhibitory neurons
    N_I = 250              # Number of excitatory neurons
    N_neurons = N_E + N_I  # Number of neurons in total
    C_E = N_E/10           # Number of excitatory synapses per neuron
    C_I = N_I/10           # Number of inhibitory synapses per neuron
    J_I = -g*J_E           # Amplitude of inhibitory postsynaptic current

    nu_ex = eta*V_th/(J_E*C_E*tau_m)
    p_rate = 1000.0*nu_ex*C_E

    nest.ResetKernel()

    # Configure kernel
    nest.SetKernelStatus({"grng_seed": 10})

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
    nest.Simulate(simulation_end)


    events_E = nest.GetStatus(spikes_E, 'events')[0]
    events_I = nest.GetStatus(spikes_I, 'events')[0]


    # Excitatory spike trains
    # Makes sure the spiketrain is added even if there are no results
    # to get a regular result
    spiketrains = []
    for sender in nodes_E[:N_rec]:
        spiketrain = events_E["times"][events_E["senders"] == sender]
        spiketrains.append(spiketrain)

    return simulation_end, spiketrains