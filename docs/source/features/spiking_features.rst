.. _spiking:

Spiking features
================



:py:class:`~uncertainpy.features.SpikingFeatures` contains a set of features
relevant for models of single
neurons that receive an external stimulus and responds by eliciting a series of
action potentials, also called spikes.
Many of these features require the start time and end time of the stimulus,
which must be returned as ``info["stimulus_start"]``
and ``info["stimulus_start"]`` in the model function.
``info`` is then used as an additional input argument in the
calculation of each feature.
``SpikingFeatures`` implements a
:py:meth:`~uncertainpy.features.SpikingFeatures.preprocess` method,
which locates spikes in the model output.

The features included in the ``SpikingFeatures`` are briefly defined
below.
This set of features was taken from the previous work of `Druckmann et al., 2007`_,
with the addition of the number of action potentials during the stimulus period.
We refer to the original publication for more detailed definitions.

.. _`Druckmann et al., 2007`: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2570085/


    1. ``nr_spikes`` -- Number of action potentials
       (during stimulus period).

    2. ``spike_rate`` -- Action potential firing rate
       (number of action potentials divided by stimulus duration).

    3. ``time_before_first_spike`` -- Time from stimulus onset to
       first elicited action potential.

    4. ``accommodation_index`` -- Accommodation index (normalized
       average difference in length of two consecutive interspike intervals).

    5. ``average_AP_overshoot`` -- Average action potential
       peak voltage.

    6. ``average_AHP_depth`` -- Average afterhyperpolarization
       depth (average minimum voltage between action potentials).

    7. ``average_AP_width`` -- Average action potential width taken
       at midpoint between the onset and peak of the action potential.



A set of standard spiking features is already included in
``SpikingFeatures``,
but the user may want to add custom features.
The :py:meth:`~uncertainpy.features.SpikingFeatures.preprocess` method changes the input given to
the feature functions,
and as such each spiking feature function has the following input arguments:

    1. The ``time`` array returned by the model simulation.
    2. An :ref:`Spikes <spikes>` object (``spikes``)
       which contain the spikes found in the model output.
    3. An ``info`` dictionary with ``info["stimulus_start"]``
       and ``info["stimulus_end"]`` set.


The ``Spikes`` object is a preprocessed version of the model output,
used as a container for ``Spike`` objects.
In turn, each ``Spike`` object contain information of a single spike.
This information includes a brief voltage trace represented by a ``time``
and a voltage (``V``) array that only includes the selected spike.
The information in ``Spikes`` is used to calculate each feature.
As an example, let us assume we want to create a feature that is the time
at which the first spike in the voltage trace ends.
Such a feature can be defined as follows::

    def first_spike_end_time(time, spikes, info):
        # Calculate the feature from the spikes object
        spike = spikes[0]              # Get the first spike
        values_feature = spike.t[-1]   # The last time point in the spike

        return None, values_feature

This feature may now be used as a feature function in the list given to the
``new_features`` argument.

From the set of both built-in and user defined features,
we may select subsets of features that we want to use in the analysis of a
model.
Let us say we are interested in how the model performs in terms of the three
features: ``nr_spikes``, ``average_AHP_depth`` and
``first_spike_end_time``.
A spiking features object that calculates these features is created by::

    features_to_run = ["nr_spikes",
                       "average_AHP_depth",
                       "first_spike_end_time"]

    features = un.SpikingFeatures(new_features=[first_spike_end_time],
                                  features_to_run=features_to_run)



API Reference
-------------

.. autoclass:: uncertainpy.features.SpikingFeatures
   :members:
   :inherited-members: