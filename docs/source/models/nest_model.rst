.. _nest_model:

NestModel
=========

`NEST`_  is a simulator for large networks of spiking neurons.
NEST models are supported through the ``NestModel`` class,
another subclass of :ref:`Model <model>`:::

    model = un.NestModel(run=nest_model_function)

``NestModel`` requires the model function to be specified through
the ``run`` argument, unlike ``NeuronModel``.
The NEST model function has the same requirements as a regular model function,
except it is restricted to return only two objects:
the final simulation time (denoted ``simulation_end``),
and a list of spike times for each neuron in the network,
which we refer to as spiketrains (denoted ``spiketrains``).

.. _NEST: http://www.nest-simulator.org/

A spike train returned by a NEST model is a set of irregularly spaced time
points where a neuron fired a spike.
NEST models therefore require postprocessing to make the model output regular.
Such a postprocessing is provided by the implemented
:py:meth:`~uncertainpy.models.NestModel.postprocess` method, which converts a spiketrain to a
list of zeros (no spike) and ones (a spike) for each time step in the simulation.
For example, if a NEST simulation returns the spiketrain ``[0, 2, 3.5]``,
it means the neuron fired three spikes occurring at
:math:`t= 0, 2, \text{and } 3.5` ms.
If the simulation have a time resolution of :math:`0.5` ms and ends
after :math:`4` ms,
``NestModel.postprocess`` returns the
postprocessed spiketrain ``[1, 0, 0, 0, 1, 0, 0, 1, 0]``,
and the postprocessed time array ``[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]``.
The final uncertainty quantification of a NEST network therefore predicts the
probability for a spike to occur at any specific time point in the simulation.
An example on how to use ``NestModel`` is found in the
:ref:`Brunel exampel <brunel>`.


API Reference
-------------

.. autoclass:: uncertainpy.models.NestModel
   :members:
   :inherited-members:
   :undoc-members:
