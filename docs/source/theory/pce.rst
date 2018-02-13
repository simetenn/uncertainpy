.. _pce:

Polynomial chaos expansions
===========================

A recent mathematical framework for estimating uncertainty is that of
polynomial chaos expansions (`Xiu and Hesthaven, 2005`_).
Polynomial chaos expansions can be seen as a subset of polynomial approximation
methods.
For a review of polynomial chaos expansions see (`Xiu, (2010)`_).
Polynomial chaos expansions are much faster than (quasi-)Monte Carlo
methods as long as the number of uncertain parameters is relatively low,
typically smaller than about twenty (`Crestaux et al.,2009`_).
This is the case for many neuroscience models,
and even for models with a higher number of uncertain parameters,
the analysis could be performed for selected subsets of the parameters.

.. _Xiu and Hesthaven, 2005: https://doi.org/10.1137/040615201
.. _Xiu, (2010): https://press.princeton.edu/titles/9229.html
.. _Crestaux et al.,2009: https://www.sciencedirect.com/science/article/pii/S0951832008002561

The general idea behind polynomial chaos expansions is to approximate the
model :math:`U` with a polynomial expansion :math:`\hat{U}`:

.. math::

    U \approx \hat{U}(\boldsymbol{x}, t, \boldsymbol{Q}) = \sum_{n=0}^{N_p - 1} c_n(\boldsymbol{x}, t) \boldsymbol{\phi}_n (\boldsymbol{Q}),

where :math:`\boldsymbol{\phi}_n` denote polynomials and :math:`c_n` denote expansion
coefficients.
The number of expansion factors :math:`N_p` is given by

.. math::

    N_p = \binom{D+p}{p},

where :math:`p` is the polynomial order.
The number of expansion coefficients in the multivariate case (:math:`D>1`) is
greater than the polynomial order.
This is because the multivariate polynomial is created by multiplying univariate
polynomials together.
The polynomials :math:`\phi_n(\boldsymbol{Q})` are chosen so they are orthogonal with respect to
the probability density function :math:`\rho_{\boldsymbol{Q}}`,
which ensures useful statistical properties.

When creating the polynomial chaos expansion,
the first step is to find the orthogonal polynomials :math:`\boldsymbol{\phi}_n`,
which in Uncertainpy is done using the so called three-term recurrence relation
(`Xiu, 2010`_).
The next step is to estimate the expansion coefficients :math:`c_n`.
The non-intrusive methods for doing this can be divided into two classes,
point-collocation methods and pseudo-spectral projection methods,
both of which are implemented in Uncertainpy.

.. _Xiu, 2010: https://press.princeton.edu/titles/9229.html


Point collocation is the default method used in Uncertainpy.
This method is based on demanding that the polynomial approximation is equal to
the model output evaluated at a set of collocation nodes drawn from the
joint probability density function :math:`\rho_{\boldsymbol{Q}}`.
This demand results in a set of linear equations for the polynomial
coefficients :math:`c_n`,
which can be solved by the use of regression methods.
The regression method used in Uncertainpy is Tikhonov regularization
(`Rifkin and Lippert, 2007`_).

.. _Rifkin and Lippert, 2007: http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf

Pseudo-spectral projection methods are based on least squares minimization in
the orthogonal polynomial space,
and finds the expansion coefficients :math:`c_n` through numerical integration.
The integration uses a quadrature scheme with weights and nodes,
and the model is evaluated at these nodes.
The quadrature method used in Uncertainpy is Leja quadrature,
with Smolyak sparse grids to reduce the number of nodes required
(`Narayan and Jakeman, 2014`_; `Smolyak, 1963`_).
Pseudo-spectral projection methods are only used in Uncertainpy when requested
by the user.

.. _Narayan and Jakeman, 2014: http://arxiv.org/abs/1404.5663
.. _Smolyak, 1963: https://www.scopus.com/record/display.uri?eid=2-s2.0-0001048298&origin=inward&txGid=581f5796dee89fce19384c4bb4f6afbc

Several of the statistical metrics of interest can be obtained directly from
the polynomial chaos expansion :math:`\hat{U}`.
The mean is simply

.. math::

    \mathbb{E}[U] \approx \mathbb{E}[\hat{U}] = c_0,

and the variance is

.. math::

    \mathbb{V}[U] \approx \mathbb{V}[\hat{U}] = \sum_{n=1}^{N_p - 1} \gamma_n c_n^2,

where :math:`\gamma_n` is a normalization factor defined as

.. math::

    \gamma_n =  \mathbb{E}\left[\boldsymbol{\phi}_n^2(\boldsymbol{Q})\right].

The first and total order Sobol indices can also be calculated directly from
the polynomial chaos expansion (`Sudret, 2008`_; `Crestaux et al.,2009`_).
On the other hand, the percentiles and prediction interval must be estimated
using :math:`\hat{U}` as a surrogate model,
and then perform the same procedure as for the (quasi-)Monte Carlo methods.

.. _Sudret, 2008: https://www.sciencedirect.com/science/article/pii/S0951832007001329
