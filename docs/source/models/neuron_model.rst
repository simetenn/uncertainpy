.. _neuron_model:

NeuronModel
===========

`NEURON`_ is a widely used simulator for multi-compartmental neural models.
Uncertainpy has support for NEURON models through the
:py:class:`~uncertainpy.model.NeuronModel` class, a subclass of :ref:`Model <model>`.
Among others, ``NeuronModel`` takes the arguments::

    model = un.NeuronModel(path="path/to/neuron_model",
                        adaptive=True,
                        stimulus_start=1000,               # ms
                        stimulus_end=1900)                 # ms

``path`` is the path to the folder where the NEURON model is saved
(the location of the ``mosinit.hoc`` file).
``adaptive`` indicates whether the NEURON model uses adaptive time steps.
``stimulus_start`` and ``stimulus_end`` denotes the start and
end time of any stimulus given to the neuron.
``NeuronModel`` loads the NEURON model from ``mosinit.hoc``,
sets the parameters of the model,
evaluates the model and returns the somatic membrane potential of the neuron.
``NeuronModel`` therefore does not require a model function.
An example of a NEURON model analysed with Uncertainpy is found in the
:ref:`interneuron example <interneuron>`.

.. _NEURON: https://www.neuron.yale.edu/neuron/

If changes are needed to the standard ``NeuronModel``,
such as measuring the voltage from other locations than the soma,
or recalculate properties after the parameters have been set,
the :ref:`Model <model>` class with an appropriate model function should be used
instead.
Alternatively,
``NeuronModel`` can be subclassed and
the existing methods customized as required.
An example of the later is shown in :ref:`/examples/bahl/ <bahl>`.




API Reference
-------------

.. autoclass:: uncertainpy.models.NeuronModel
   :members:
   :inherited-members:
   :undoc-members: