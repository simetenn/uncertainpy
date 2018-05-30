.. _features:

Features
========

The activity of a biological system typically varies between recordings,
even if the experimental conditions are maintained constant to the highest
degree possible.
Since the experimental data displays such variation,
it is often meaningless (or even misguiding) to base the success of a
computational model on a direct point-to-point comparison between the
experimental data and model output (`Druckmann et al., 2007`_; `Van Geit et al., 2008`_).
A common modeling practice is therefore to rather have the model reproduce
essential features of the experimentally observed dynamics,
such as the action potential shape, or action potential firing rate
(`Druckmann et al., 2007`_).
Such features are typically more robust between different experimental
measurements, or between different model simulations,
than the raw data or raw model output,
at least if sensible features have been chosen.

.. _Druckmann et al., 2007: http://journal.frontiersin.org/article/10.3389/neuro.01.1.1.001.2007/abstract
.. _Van Geit et al., 2008: https://link.springer.com/article/10.1007/s00422-008-0257-6

Uncertainpy takes this aspect of neural modeling into account,
and is constructed so it can extract a set of features relevant for various
common model types in neuroscience from the raw model output.
Examples include the action potential shape in single neuron models,
or the average interspike interval in network models.
If we give the ``features`` argument to
:ref:`UncertaintyQuantification <UncertaintyQuantification>`,
Uncertainpy will perform uncertainty quantification and sensitivity analysis
of the given features,
in addition to the analysis of the "raw" output data.
The value of feature based analysis is illustrated in the two examples on
:ref:`a multi-compartment model of a thalamic interneuron <interneuron>` and
:ref:`a sparsely connected recurrent network <brunel>`.

The main class is :ref:`Features <main_features>`.
This class does not implement any specific features itself, but
contain all common methods used by features.
It is also used when creating custom features.
Three sets of features comes pre-defined with Uncertainpy. Two sets of features
for spiking models that returns voltage traces: :ref:`SpikingFeatures <spiking>`
and :ref:`EfelFeatures <efel>`.
And one set of features for network models that return spiketrains :ref:`NetworkFeatures <network>`
Then there are two general classes
for spiking (:ref:`GeneralSpikingFeatures <general_spiking>`) and network features
(:ref:`GeneralNetworkFeatures <general_network>`) that implements common methods
used by the two spiking features and network features respectively.
These classes does not implement any specific models themselves.


.. toctree::
    :maxdepth: 1

    features/master_features
    features/spiking_features
    features/spikes
    features/efel_features
    features/network_features
    features/general_network_features
    features/general_spiking_features