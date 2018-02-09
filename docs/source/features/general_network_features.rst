.. _general_network:

GeneralNetworkFeatures
======================

:py:class:`~uncertainpy.features.GeneralNetworkFeatures` implements the
preprocessing of spiketrains, and create `NEO`_ spiketrains,
but does not implement any features in itself.
The :py:meth:`~uncertainpy.features.GeneralNetworkFeatures.preprocess` method
changes the input given to
the feature functions,
and as such each network feature function has the following input arguments:

1. End time of the simulation (``end_time``).
2. A list of `NEO`_  spiketrains (``spiketrains``).

.. _NEO: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3930095/

API Reference
-------------

.. autoclass:: uncertainpy.features.GeneralNetworkFeatures
   :members:
   :inherited-members: