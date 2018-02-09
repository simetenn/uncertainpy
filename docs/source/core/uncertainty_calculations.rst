UncertaintyCalculations
=======================


:py:class:`~uncertainpy.core.UncertaintyCalculations` is the class responsible for
performing the uncertainty calculations.
Here we explain how they are performed as well as well as which options the user
have to customize the calculations
An insight into how the calculations are performed
is not required to use Uncertainpy.
In most cases, the default settings works fine.
In addition to the customization options shown below,
Uncertainpy has support for implementing entirely custom
uncertainty quantification and sensitivity analysis methods.
This is only recommended for expert users,
as knowledge of both Uncertainpy and uncertainty quantification is needed.


Quasi-Monte Carlo method
------------------------

To use the quasi-Monte Carlo method, we call
:py:meth:`~uncertainpy.UncertaintyQuantification.quantify` with
``method="mc"``, and the optional argument ``nr_mc_samples``::

    UQ.quantify(
        method="mc",
        nr_mc_samples=10**3,
    )

By default, the quasi-Monte Carlo method quasi-randomly draws `1000`
parameter samples from the joint multivariate probability distribution of the
parameters :math:`\rho_{\boldsymbol{Q}}` using Hammersley sampling (`Hammersley, 1960`_).
As the name indicates, the number of samples is specified by the
``nr_mc_samples`` argument.
The model is evaluated for each of these parameter samples,
and features are calculated for each model evaluation (when applicable).
To speed up the calculations,
Uncertainpy uses the multiprocess Python package
(`McKerns et al., 2012`_) to perform this step in parallel.
When model and feature calculations are done,
Uncertainpy calculates the mean, variance,
and 5th and 95th percentile (which gives the `90\%` prediction interval)
for the model output as well as for each feature.

.. _McKerns et al., 2012: https://arxiv.org/pdf/1202.1056.pdf
.. _Hammersley, 1960: http://onlinelibrary.wiley.com/doi/10.1111/j.1749-6632.1960.tb42846.x/pdf

Polynomial chaos expansions
---------------------------

To use polynomial chaos expansions we use :py:meth:`~uncertainpy.UncertaintyQuantification.quantify`
with the argument ``method="pc"``,
which takes a set of optional arguments (default are values specified)::

    UQ.quantify(
        method="pc",
        pc_method="collocation",
        rosenblatt=False,
        polynomial_order=3,
        nr_collocation_nodes=None,
        quadrature_order=None,
        nr_pc_mc_samples=10**4,
    )

As previously mentioned, Uncertainpy allows the user to select between point
collocation (``pc_method="collocation"``)
and pseudo-spectral projections (``pc_method="spectral"``).
The goal is to create separate polynomial chaos expansions `\hat{U}` for the
model and each feature.
In both methods,
Uncertainpy creates the orthogonal polynomial :math:`\boldsymbol{\phi}_n` using the
three-term recurrence relation and :math:`\rho_{\boldsymbol{Q}}`.
Uncertainpy uses a third order polynomial expansion,
changed with ``polynomial_order``.
The polynomial :math:`\boldsymbol{\phi}_n` is shared between the model and all features,
since they have the same uncertain input parameters,
and therefore the same :math:`\rho_{\boldsymbol{Q}}`.
Only the polynomial coefficients :math:`c_n` differ between the model and each feature.

The two polynomial chaos methods differ in terms of how they calculate :math:`c_n`.
For point collocation Uncertainpy uses :math:`2(N_p + 1)` collocation nodes,
as recommended by (`Hosder et al., 2007`_),
where `N_p` is the number of polynomial chaos expansion factors.
The number of collocation nodes can be customized with
``nr_collocation_nodes``,
but the new number of nodes must be chosen carefully.
The collocation nodes are sampled from :math:`\rho_{\boldsymbol{Q}}` using
Hammersley sampling (`Hammersley, 1960`_).
The model and features are calculated for each of the collocation nodes.
As with the quasi-Monte Carlo method, this step is performed in parallel.
The polynomial coefficients :math:`c_n` are calculated
using Tikhonov regularization (`Rifkin and Lipert, 2007`_) from the model and feature
results.

.. _Hosder et al., 2007: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.610&rep=rep1&type=pdf
.. _Rifkin and Lipert, 2007: http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf
.. _Narayan and Jakeman, 2014: http://epubs.siam.org/doi/pdf/10.1137/140966368
.. _Smolyak, 1963: https://www.scopus.com/record/display.uri?eid=2-s2.0-0001048298&origin=inward&txGid=909fc4b912013bd67236ad5d9d593074

For the pseudo-spectral projection,
Uncertainpy chooses nodes and weights using a quadrature scheme,
instead of choosing nodes from :math:`\rho_{\boldsymbol{Q}}`.
The quadrature scheme used is Leja quadrature with a Smolyak sparse grid
(`Narayan and Jakeman, 2014`_; `Smolyak, 1963`_).
The Leja quadrature is of order two greater than the polynomial
order,
but can be changed with ``quadrature_order``.
The model and features are calculated for each of the quadrature nodes.
As before, this step is performed in parallel.
The polynomial coefficients :math:`c_n` are then calculated from the quadrature nodes,
weights, and model and feature results.

When Uncertainpy has derived :math:`\hat{U}` for the model and features,
it uses :math:`\hat{U}` to compute the mean, variance,
and the first and total order Sobol indices.
The first and total order Sobol indices are also summed and normalized.
Finally, Uncertainpy uses :math:`\hat{U}` as a surrogate model,
and performs a quasi-Monte Carlo method with Hammersley sampling and
``nr_pc_mc_samples=10**4``  samples to find the
5th and 95th percentiles.

If the model parameters have a dependent joint multivariate distribution,
the Rosenblatt transformation must be used by setting
``rosenblatt=True``.
To perform the transformation Uncertainpy chooses
:math:`\rho_{\boldsymbol{R}}` to be a multivariate independent normal distribution,
which is used instead of :math:`\rho_{\boldsymbol{Q}}` to perform the polynomial chaos expansions.
Both the point collocation method and the pseudo-spectral method are performed
as described above.
The only difference is that we use :math:`\rho_{\boldsymbol{R}}` instead of :math:`\rho_{\boldsymbol{Q}}`,
and use the Rosenblatt transformation to transform the selected nodes
from :math:`\boldsymbol{R}` to :math:`\boldsymbol{Q}`, before they are used in the model evaluation.


API Reference
-------------

.. autoclass:: uncertainpy.core.UncertaintyCalculations
   :members:
   :inherited-members:
   :undoc-members: