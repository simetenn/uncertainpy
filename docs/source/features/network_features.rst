.. _network:

NetworkFeatures
===============

:py:class:`~uncertainpy.features.NetworkFeatures` contains a set of features
relevant for the output of NEST network
models and are calculated using the `Elephant software`_.
The implemented features are:

.. _Elephant software: http://neuralensemble.org/elephant/

1. ``mean_firing_rate`` -- Mean firing rate (averaged over all recorded neurons).
2. ``instantaneous_rate`` -- Instantaneous firing rate (averaged over all recorded neurons within a small time window).
3. ``mean_isi`` -- Mean interspike interval (averaged over all recorded neurons).
4. ``cv`` -- Coefficient of variation of the interspike interval (for a single recorded neuron).
5. ``mean_cv`` -- Mean coefficient of variation of the interspike interval (averaged over all recorded neurons).
6. ``lv`` -- Local variation (variability of interspike intervals for a single recorded neuron).
7. ``mean_lv`` -- Mean local variation (variability of interspike intervals averaged over all recorded neurons).
8. ``fanofactor`` -- Fanofactor (variability of spiketrains).
9. ``victor_purpura_dist`` -- Victor purpura distance (spiketrain dissimilarity between two recorded neurons).
10. ``van_rossum_dist`` -- Van rossum distance (spiketrain dissimilarity between two recorded neurons).
11. ``binned_isi`` -- Histogram of the interspike intervals (for all recorded neurons).
12. ``corrcoef`` -- Pairwise Pearson's correlation coefficients (between the spiketrains of two recorded neurons).
13. ``covariance`` -- Covariance (between the spiketrains of two recorded neurons).


The use of the ``NetworkFeatures`` class in Uncertainpy follows the
same logic as the use of the other feature classes,
and custom features can easily be included.
As with :ref:`SpikingFeatures <spiking>`,
``NetworkFeatures`` implements a :py:meth:`~uncertainpy.features.NetworkFeatures.preprocess` method.
This ``preprocess`` returns the following objects:

1. End time of the simulation (``end_time``).
2. A list of `NEO`_  spiketrains (``spiketrains``).

.. _NEO: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3930095/

Each feature function therefore require the same objects as input arguments.
Note that a ``info`` object is not used.


API Reference
-------------

.. autoclass:: uncertainpy.features.NetworkFeatures
   :members:
   :inherited-members: