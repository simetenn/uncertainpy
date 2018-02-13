.. _uq:

Uncertainty quantification
==========================

The goal of an uncertainty quantification is to describe the unknown
distribution of the model output :math:`\rho_Y` through statistical metrics.
The two most common statistical metrics used in this context are the
mean :math:`\mathbb{E}` (also called the expectation value)
and the variance :math:`\mathbb{V}`.
The mean is defined as:

.. math::

    \mathbb{E}[Y] = \int_{\Omega_Y} y\rho_Y(y)dy,

and tells us the expected value of the model output :math:`Y`.
The variance is defined as:

.. math::

    \mathbb{V}[Y] = \int_{\Omega_Y} {\left(y - \mathbb{E}[Y]\right)}^2\rho_Y(y)dy,

and tells us how much the output varies around the mean.

Another useful metric is the :math:`(100\cdot x)`-th percentile :math:`P_x` of :math:`Y`,
which defines a value below which :math:`100 \cdot x` percent of the simulation
outputs are located.
For example, 5% of the simulations of a model will give an output lower than
the :math:`5`-th percentile.
The :math:`(100\cdot x)`-th percentile is defined as:

.. math::

    x = \int_{-\infty}^{P_x}\rho_Y(y)dy.


We can combine two percentiles to create a prediction interval :math:`I_x`,
which is a range of values such that a :math:`100\cdot x` percentage of the outputs
:math:`Y` occur within this range:

.. math::

    I_x = \left[P_{(x/2)}, P_{(1-x/2)}\right]. \label{eq:prediction}

The :math:`90\%` prediction interval gives us the interval within :math:`90\%` of the :math:`Y` outcomes occur,
which also means that :math:`5\%` of the outcomes are above and :math:`5\%` below this interval.
