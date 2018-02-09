.. _bahl:

A layer 5 pyramidal neuron implemented with NEURON
==================================================


In this example we illustrate how we can subclass a :ref:`NeuronModel <neuron_model>`
to customize the methods.
We select a set of reduced models of layer 5 pyramidal neurons (`Bahl et al., 2012`_).
The code for this example is found in
:download:`/examples/bahl/uq_bahl.py <../../../examples/bahl/uq_bahl.py>`.
To be able to run this example you require both the `NEURON`_ simulator,
as well as the layer 5 pyramidal neuron model saved in the folder ``/bahl_model/``.

.. _Bahl et al., 2012: https://www.sciencedirect.com/science/article/pii/S016502701200129X?via%3Dihub
.. _NEURON: https://www.neuron.yale.edu/neuron/


Since the model is implemented in NEURON, we use the :ref:`NeuronModel <neuron_model>`.
The problem is that this model require us to recalculate certain properties of
the model after the parameters have been set.
We therefore have to make change to the ``NeuronModel`` class so we recalculate
these properties.
The standard :py:func:`~uncertainpy.models.NeuronModel.run` method implemented in
``NeuronModel`` calls
:py:func:`~uncertainpy.models.NeuronModel.set_parameters` to set the parameters.
We therefore only need to change this method in the ``NeuronModel``.
First we subclass ``NeuronModel``. For ease of use, we hardcode in the path to
the Bahl model.

.. literalinclude:: ../../../examples/bahl/uq_bahl.py
    :language: python
    :lines: 4-11

We then implement a new ``set_parameters`` method, that recalculates the required
properties after the parameters have been set.

.. literalinclude:: ../../../examples/bahl/uq_bahl.py
    :language: python
    :lines: 13-21

Now we can initialize our new model.

.. literalinclude:: ../../../examples/bahl/uq_bahl.py
    :language: python
    :lines: 24-25

We can then create the uncertain parameters, which we here set to be ``"e_pas"``
and ``"apical Ra"``.
Here we do not create a :ref:`Parameter <Parameters class>` object,
but use the parameter list directly, to show that this option exists.

.. literalinclude:: ../../../examples/bahl/uq_bahl.py
    :language: python
    :lines: 27-29

The we use :ref:`SpikingFeatures <spiking>`.

.. literalinclude:: ../../../examples/bahl/uq_bahl.py
    :language: python
    :lines: 31-32

Lastly we set up and perform the uncertainty quantification and
sensitivity analysis.

.. literalinclude:: ../../../examples/bahl/uq_bahl.py
    :language: python
    :lines: 34-38

The complete code becomes:

.. literalinclude:: ../../../examples/bahl/uq_bahl.py
    :language: python