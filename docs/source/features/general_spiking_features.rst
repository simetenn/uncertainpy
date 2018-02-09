.. _general_spiking:

GeneralSpikingFeatures
======================

:py:class:`~uncertainpy.features.GeneralSpikingFeatures` implements the
preprocessing of voltage traces, and locate spikes in the voltage traces,
but does not implement any features in itself.
The :py:meth:`~uncertainpy.features.GeneralSpikingFeatures.preprocess` method
changes the input given to the feature functions,
and as such each spiking feature function has the following input arguments:

1. The ``time`` array returned by the model simulation.
2. An :ref:`Spikes <spikes>` object (``spikes``) which contain the spikes found in the model output.
3. An ``info`` dictionary with ``info["stimulus_start"]`` and ``info["stimulus_end"]`` set.


API Reference
-------------

.. autoclass:: uncertainpy.features.GeneralSpikingFeatures
   :members:
   :inherited-members: