.. _features:

Features
========

It is often more meaningful to examine the uncertainty in features
of the model output,
than to base the analysis directly on a point-to-point comparison of the "raw"
output data (e.g. a voltage trace).
Upon user request,
Uncertainpy can identify and extract features of the model output.
If we give the ``features`` argument to
:ref:`UncertaintyQuantification <UncertaintyQuantification>`,
Uncertainpy will perform uncertainty quantification and sensitivity analysis
of the given features,
in addition to the analysis of the "raw" output data.


The main class is :ref:`Features <main_features>`.
This class does not implement any specific features itself, but
contain all common methods used by features.
It is also used when creating custom features.


Three sets of features comes pre-defined with Uncertainpy. Two sets of features
for spiking models that returns voltage traces: :ref:`SpikingFeatures <spiking>`
and :ref:`EfelFeatures <efel>`.
And one set of features for network models that return spiketrains :ref:`NetworkFeatures <network>`


Then there are two general classes
for spiking (:ref:`GeneralSpikingFeatures <_general_spiking>`) and network features
(:ref:`GeneralNetworkFeatures <general_network>`) that implements common methods
used by the two spiking features and network features respectively.
These classes does not implement any specific models themselves.


.. toctree::
    :maxdepth: 1

    features/master_features
    features/spikes
    features/general_spiking_features
    features/spiking_features
    features/efel_features
    features/general_network_features
    features/network_features
