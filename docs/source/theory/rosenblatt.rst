.. _rosenblatt:

The Rosenblatt transformation
=============================

One of the underlying assumptions when creating the polynomial chaos expansion is that the model
parameters are independent.
However, dependent parameters in neuroscience models are quite common.
Fortunately, models containing dependent parameters can be analyzed
with Uncertainpy by the aid of the Rosenblatt transformation from Chaospy.
The idea is to use the Rosenblatt transformation to create a reformulated model
:math:`\widetilde{U}(\boldsymbol{x}, t, \boldsymbol{R})`, that
takes an arbitrary independent parameter set :math:`\boldsymbol{R}` as input,
instead of the original dependent parameter set :math:`\boldsymbol{Q}`.
We use the Rosenblatt transformation to transform from :math:`\boldsymbol{R}` to
:math:`\boldsymbol{Q}`, which makes it so :math:`\widetilde{U}` give the same output
(and statistics) as the original model:

.. math::
    \widetilde{U}(\boldsymbol{x}, t, \boldsymbol{R}) = U(\boldsymbol{x}, t, \boldsymbol{Q}).

We can then perform polynomial chaos expansion as normal on the reformulated model,
since it has independent parameters.



The Rosenblatt transformation :math:`T_{\boldsymbol{Q}}` transforms the
random variable :math:`\boldsymbol{Q}` to
the random variable :math:`\boldsymbol{H}`,
which in a statistical context behaves as if it were drawn uniformly from the unit
hypercube :math:`{[0, 1]}^D`.

.. math::

    T_{\boldsymbol{Q}}(\boldsymbol{Q}) = \boldsymbol{H}.

Here, :math:`T_{\boldsymbol{Q}}` denotes a Rosenblatt transformation which is uniquely defined by
:math:`\rho_Q` (the probability distribution of :math:`\boldsymbol{Q}`).
We can use the Rosenblatt transformation to transform from :math:`\boldsymbol{R}` to :math:`\boldsymbol{Q}`
(through :math:`\boldsymbol{H}`) to regain our original parameters:

.. math::

    T_{\boldsymbol{Q}}(\boldsymbol{Q}) &= \boldsymbol{H} = T_{\boldsymbol{R}}(\boldsymbol{R}) \\
            \boldsymbol{Q} &= T_{\boldsymbol{Q}}^{-1}(T_{\boldsymbol{R}}(\boldsymbol{R})).


Using this relation between :math:`\boldsymbol{R}` and :math:`\boldsymbol{Q}` in we can
reformulate our model to take :math:`\boldsymbol{R}` as input,
but still give the same results:

.. math::

    U(\boldsymbol{x}, t, \boldsymbol{Q})
    = U(\boldsymbol{x}, t, T_{\boldsymbol{Q}}^{-1}(T_{\boldsymbol{R}}(\boldsymbol{R})))
    = \widetilde{U}(\boldsymbol{x}, t, \boldsymbol{R}).

The statistical analysis can now be performed on this reformulated model
:math:`\widetilde{U}` as before.