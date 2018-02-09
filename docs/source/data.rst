.. _data:

Data
====


Uncertainpy stores all results from the uncertainty quantification and
sensitivity analysis in ``UncertaintyQuantification.data``,
as a ``Data`` object.
The ``Data`` class works similarly to a Python dictionary.
The name of the model or feature is the key,
while the values are ``DataFeature`` objects that stores each
statistical metric in in the table below as attributes.
Results can be saved and loaded through
``Data.save`` and ``Data.load``.


================================================  ========================    ========================
Calculated statistical metric                     Symbol                      Variable
================================================  ========================    ========================
Model and feature evaluations                     :math:`U`                   ``values``
Model and feature times                           :math:`t`                   ``time``
Mean                                              :math:`\mathbb{E}`          ``mean``
Variance                                          :math:`\mathbb{V}`          ``variance``
5th percentile                                    :math:`P_{5}`               ``percentile_5``
95th percentile                                   :math:`P_{95}`              ``percentile_95``
First order Sobol indices                         :math:`S`                   ``sobol_first``
Total order Sobol indices                         :math:`S_T`                 ``sobol_total``
Normalized sum of first order Sobol indices       :math:`\widehat{S}`         ``sobol_first_sum``
Normalized sum of total order Sobol indices       :math:`\widehat{S}_{T}`     ``sobol_total_sum``
================================================  ========================    ========================


An example: if we have performed uncertainty quantification of a spiking
neuron model with the number of spikes as one of the features,
we get the variance of the number of spikes by typing:::

    data = un.Data()
    data.load("filename")
    variance = data["nr_spikes"].variance


API reference
-------------

.. toctree::
    :maxdepth: 1

    data/data_feature
    data/data