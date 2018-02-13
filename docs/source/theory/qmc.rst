.. _qmc:

(Quasi-)Monte Carlo methods
===========================

A typical way to obtain the statistical metrics mentioned above is to use
(quasi-)Monte Carlo methods.
We give a brief overview of these methods here,
for more comprehensive reviews see `Lemieux, (2009)`_; `Rubinstein and Kroese (2016)`_.

.. _Lemieux, (2009): http://www.springer.com/us/book/9780387781648
.. _Rubinstein and Kroese (2016): http://onlinelibrary.wiley.com/book/10.1002/9781118631980

The general idea behind the standard Monte Carlo method is quite simple.
A set of parameters is pseudo-randomly drawn from the joint multivariate probability
density function :math:`\rho_{\boldsymbol{Q}}` of the parameters.
The model is then evaluated for the sampled parameter set.
This process is repeated thousand of times,
and statistical metrics such as the mean and variance are computed for the
resulting series of model outputs.
The problem with the standard Monte Carlo method is that a very high number of
model evaluations is required to get reliable statistics.
If the model is computationally expensive,
the Monte Carlo method may require insurmountable computer power.

Quasi-Monte Carlo methods improve upon the standard Monte Carlo method by using
variance-reduction techniques to reduce the number of model evaluations needed.
These methods are based on increasing the coverage of the sampled parameter
space by distributing the samples more evenly.
Fewer samples are then required to get a given accuracy.
Instead of pseudo-randomly selecting parameters from :math:`\rho_{\boldsymbol{Q}}`,
the samples are selected using a low-discrepancy sequence such as the
Hammersley sequence (`Hammersley, 1960`_).
Quasi-Monte Carlo methods are faster than the Monte Carlo method,
as long as the number of uncertain parameters is sufficiently small
(`Lemieux, 2009`_).

.. _Hammersley, 1960: http://dx.doi.org/10.1111/j.1749-6632.1960.tb42846.x
.. _Lemieux, 2009: http://www.springer.com/us/book/9780387781648



Uncertainpy allows quasi-Monte Carlo methods to be used to compute the
statistical metrics.
When this option is chosen, the metrics are computed as follows.
With :math:`N` model evaluations,
which gives the results :math:`\boldsymbol{Y} = [Y_1, Y_2, \ldots, Y_N]`,
the mean is given by

.. math::

    \mathbb{E}[\boldsymbol{Y}] \approx \frac{1}{N}\sum_{i=1}^{N} Y_i,

and the variance by

.. math::

    \mathbb{V}[\boldsymbol{Y}] \approx \frac{1}{N-1}\sum_{i=1}^{N} {\left(Y_i - \mathbb{E}[Y]\right)}^2.

Prediction intervals are found by sorting the model evaluations
:math:`\boldsymbol{Y}` in an increasing order,
and then find the :math:`(100\cdot x/2)`-th and :math:`(100\cdot (1 - x/2))`-th percentiles.
The sensitivity analysis in Uncertainpy is based on polynomial chaos expansions
(see below),
and Uncertainpy does currently not support calculation of Sobol indices from
(quasi-)Monte Carlo methods,
although methods for this are available in the literature (`Saltelli et al., 2010`_).


.. _Saltelli et al., 2010: http://dx.doi.org/10.1016/j.cpc.2009.09.018