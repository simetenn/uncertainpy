.. _brunel:

A sparsely connected recurrent network using Nest
=================================================


In the last case study,
we use Uncertainpy to perform a feature based analysis of the sparsely
connected recurrent network by `Brunel (2000)`_.
We implement the Brunel network using `NEST`_ inside a Python function,
and create :math:`10000` inhibitory and :math:`2500` excitatory neurons.
We record the output from :math:`20` of the excitatory neurons,
and simulate the network for :math:`1000` ms.
This is the values used to create the results in the Uncertainpy paper.
If you want to just test the network, we recommend reducing the model to
:math:`2000` inhibitory and :math:`500` excitatory neurons,
and only simulate the network for :math:`100` ms.
To be able to run this example you require NEST to be anle to run the model and
``elephant``, ``neo``, and ``quantities`` to be able to use the network features.

.. _Brunel (2000): https://web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf
.. _NEST: http://www.nest-simulator.org/


We want to use :ref:`NestModel <nest>` to create our model.
``NestModel`` requires the model function to be specified through
the ``run`` argument, unlike ``NeuronModel``.
The NEST model function has the same requirements as a regular model function,
except it is restricted to return only two objects:
the final simulation time (``simulation_end``),
and a list of spike times for each neuron in the network (``spiketrains``).
``NestModel`` then postproccess this result for us to a regular result.
The final uncertainty quantification of a NEST network therefore predicts the
probability for a spike to occur at any specific time point in the simulation.
We implement the Brunel network as such a function
(found in :download:`/examples/brunel/brunel.py <../../../examples/brunel/brunel.py>`):

.. literalinclude:: ../../../examples/brunel/brunel.py
    :language: python


And use it to create our model (example found in
:download:`/examples/brunel/uq_brunel.py <../../../examples/brunel/uq_brunel.py>`):

.. literalinclude:: ../../../examples/brunel/uq_brunel.py
    :language: python
    :lines: 6-7


The Brunel model has four uncertain parameters:

1. the external rate (:math:`\nu_\mathrm{ext}`) relative to threshold rate
   (:math:`\nu_\mathrm{thr}`) given as :math:`\eta = \nu_\mathrm{ext}/\nu_\mathrm{thr}`,
2. the relative strength of the inhibitory synapses :math:`g`,
3. the synaptic delay :math:`D`, and
4. the amplitude of excitatory postsynaptic current :math:`J_e`.

Depending on the parameterizations of the model,
the Brunel network may be in several different activity states.
For the current example,
we limit our analysis to two of these states.
We create two sets of parameters, one for each of two states,
and assume the parameter uncertainties are characterized by uniform probability
distributions within the ranges below:

===============    =====================   =====================    ============    ==================================================
Parameter          Range SR                Range AI                 Variable        Meaning
===============    =====================   =====================    ============    ==================================================
:math:`\eta`       :math:`[1.5, 3.5]`      :math:`[1.5, 3.5]`       ``eta``         External rate relative to threshold rate
:math:`g`          :math:`[1, 3]`          :math:`[5, 8]`           ``g``           Relative strength of inhibitory synapses
:math:`D`          :math:`[1.5, 3]`        :math:`[1.5, 3]`         ``delay``       Synaptic delay (ms)
:math:`J_e`        :math:`[0.05, 0.15]`    :math:`[0.05, 0.15]`     ``J_e``         Amplitude excitatory postsynaptic current (mV)
===============    =====================   =====================    ============    ==================================================


These ranges correspond to the synchronous regular (SR) state,
where the neurons are almost completely synchronized,
and the asynchronous irregular (AI) state,
where the neurons fire individually at low rates.
We create two sets of parameters, one for each state:

.. literalinclude:: ../../../examples/brunel/uq_brunel.py
    :language: python
    :lines: 10-22


We use the features in :ref:`NetworkFeatures <network>` to
examine features of the Brunel network.

.. literalinclude:: ../../../examples/brunel/uq_brunel.py
    :language: python
    :lines: 24-25

We set up the problems with the SR parameter set and use polynomial chaos with
point collocation to perform the uncertainty quantification and sensitivity
analysis.
We specify a filename for the data, and a folder where to save the figures, to
keep the results from the AI and SR state separated.

.. literalinclude:: ../../../examples/brunel/uq_brunel.py
    :language: python
    :lines: 27-35

We then change the parameters, and perform the uncertainty quantification and
sensitivity analysis for the new set of parameters,
again specifying a filename and figure folder.

.. literalinclude:: ../../../examples/brunel/uq_brunel.py
    :language: python
    :lines: 37-43

The complete code is:

.. literalinclude:: ../../../examples/brunel/uq_brunel.py
    :language: python

