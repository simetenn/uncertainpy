.. _coffee_dependent:

A cooling coffee cup model with dependent parameters
====================================================

Here we show an example (found in
:download:`examples/coffee_dependent/uq_coffee_dependent_function.py <../../../examples/coffee_dependent/uq_coffee_dependent_function.py>`)
where we examine a cooling coffee cup model with dependent parameters.
We modify the :ref:`simple cooling coffee cup <coffee_cup>` model by introducing two auxillary variables
:math:`\alpha` and :math:`\hat{\kappa}`:

.. math::

    \kappa = \alpha\hat{\kappa}

to get:

.. math::

    \frac{dT(t)}{dt} = -\alpha\hat{\kappa}\left(T(t) - T_{env}\right).

The auxillary variables are made dependent by requiring that the model should be
identical to the original model.
We assume that :math:`\alpha` is an uncertain scaling factor:

.. math::

    \alpha = \mathrm{Uniform}(0.5, 1.5),

and set:

.. math::

    \hat{\kappa} = \frac{\kappa}{\alpha}.

Which gives us the following distributions:


.. math::
    \alpha &= \mathrm{Uniform}(0.5, 1.5)

    \hat{\kappa} &= \frac{\mathrm{Uniform}(0.025, 0.075)}{\alpha}

    T_{env} &= \mathrm{Uniform}(15, 25).


With Chaospy we can create these dependencies using arithmetic operators:

.. literalinclude:: ../../../examples/coffee_dependent/uq_coffee_dependent_function.py
    :language: python
    :lines: 27-36

To be able to use polynomial chaos methods when we have dependent parameters
we need to use the Rosenblatt transformation, which we enable by::

    UQ.quantify(rosenblatt=True)

The complete code example become:

.. literalinclude:: ../../../examples/coffee_dependent/uq_coffee_dependent_function.py
    :language: python

In this case,
the distribution we assign to :math:`\alpha` does not matter for the end result,
as the distribution for :math:`\hat{\kappa}` will be scaled accordingly.
Using the Rosenblatt transformation,
an uncertainty quantification and sensitivity analysis of the
dependent coffee cup model therefore returns the same results as seen in
the simple coffee cup model,
where the role of the original :math:`\kappa` is taken over by :math:`\hat{\kappa}`,
while the sensitivity to the additional parameter :math:`\alpha` becomes strictly zero.