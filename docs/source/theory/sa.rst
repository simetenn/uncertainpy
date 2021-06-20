.. _sa:

Sensitivity analysis
====================

Sensitivity analysis quantifies how much of the uncertainty in the model output
each uncertain parameter is responsible for.
It is the computational equivalent of analysis of variance (ANOVA) performed by
experimentalists (`Archer et al., 1997`_).
For a review of different sensitivity analysis methods,
see `Hamby (1994)`_; `Borgonovo and Plischke (2016)`_. Several different sensitivity measures exist,
but Uncertainpy uses the commonly used Sobol sensitivity indices (`Sobol, 1990`_).
The Sobol sensitivity indices quantify how much of the variance in the model
output each uncertain parameter is responsible for.
If a parameter has a low sensitivity index,
variations of this parameter results in comparatively small variations in the
final model output.
On the other hand, if a parameter has a high sensitivity index,
a change in this parameter leads to a dramatic change in the model output.

.. _Archer et al., 1997: http://www.tandfonline.com/doi/abs/10.1080/00949659708811825
.. _Hamby (1994): https://link.springer.com/article/10.1007/BF00547132
.. _Borgonovo and Plischke (2016): http://dx.doi.org/10.1016/j.ejor.2015.06.032
.. _Sobol, 1990: http://www.mathnet.ru/eng/mm2320

A sensitivity analysis provides a better understanding of the relationship
between the parameters and output of a model.
This can be useful in a model reduction context.
For example, a parameter with a low sensitivity index can essentially be set to
any fixed value (within the explored distribution),
without affecting the variance of the model much.
In some cases, such an analysis can justify leaving out entire mechanisms from
a model.
For example, if a single neuron model is insensitive to the conductance of a
given ion channel :math:`g_x`,
this ion channel could possibly be removed from the model without changing the
model behavior much.
Additionally, a model-based sensitivity analysis can guide the experimental focus,
so that special care is taken to obtain accurate measures of parameters with
high sensitivity indices,
while more crude measures are acceptable for parameters with low sensitivity
indices.

There exist several types of Sobol indices.
The first order Sobol sensitivity index :math:`S` measures the direct effect each
parameter has on the variance of the model:

.. math::

    S_i = \frac{\mathbb{V}[\mathbb{E}[Y | Q_i]]}{\mathbb{V}[Y]}.

Here, :math:`\mathbb{E}[{Y | Q_i}]` denotes the expected value of the output :math:`Y` when parameter
:math:`Q_i` is fixed.
The first order Sobol sensitivity index tells us the expected reduction in the
variance of the model when we fix parameter :math:`Q_i`.
The sum of the first order Sobol sensitivity indices can not exceed one
(`Glen and Isaacs, 2012`_).

.. _Glen and Isaacs, 2012: http://dx.doi.org/10.1016/j.envsoft.2012.03.014

Higher order sobol indices exist,
and give the sensitivity due interactions between a parameter :math:`Q_i` and various
other parameters.
It is customary to only calculate the first and total order indices
(`Saltelli et al., 2010`_).
The total Sobol sensitivity index :math:`S_{Ti}` includes the sensitivity of both
first order effects as well as the sensitivity due to interactions (covariance)
between a given parameter :math:`Q_i` and all other parameters (`Homma and Saltelli, 1996`_).
It is defined as:

.. math::

    S_{Ti} = 1 - \frac{\mathbb{V}[\mathbb{E}[Y | Q_{-i}]]}{\mathbb{V}[Y]},

where :math:`Q_{-i}` denotes all uncertain parameters except :math:`Q_{i}`.
The sum of the total Sobol sensitivity indices is equal to or greater than one
(`Glen and Isaacs, 2012`_).
If no higher order interactions are present,
the sum of both the first and total order sobol indices are equal to one.

.. _Saltelli et al., 2010: http://dx.doi.org/10.1016/j.cpc.2009.09.018
.. _Homma and Saltelli, 1996: http://www.sciencedirect.com/science/article/pii/0951832096000026

We might want to compare Sobol indices across different features
(see in :ref:`Features <features>`).
This can be problematic when we have features with different number of output
dimensions.
In the case of a zero dimensional output the Sobol indices is a single number,
while for a one dimensional output we get Sobol indices for each point in time.
To better be able to compare the Sobol indices across such features,
we therefore calculate the average of both the first order Sobol
indices :math:`\widehat{S}`,
and the total order Sobol indices :math:`\widehat{S}_{T}`.
