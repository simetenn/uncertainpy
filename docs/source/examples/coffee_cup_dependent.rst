.. _coffee_dependent:

A cooling coffee cup model with dependent parameters
====================================================

Here we show an example (found in ``examples/coffee_dependent``) where we
examine a cooling coffee cup model with dependent parameters.
We modify the :ref:`coffee cup <coffee_cup>` model by introducing two auxillary variables
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


With Chaospy we can create these dependencies using arithmetic operators::

    # Create the distributions
    alpha_dist = cp.Uniform(0.5, 1.5)
    kappa_hat_dist = cp.Uniform(0.025, 0.075)/alpha_dist
    T_env_dist = cp.Uniform(15, 25)

    # Define a parameter list and use it to create the Parameters
    parameter_list = [["alpha", None, alpha_dist],
                      ["kappa_hat", None, kappa_hat_dist],
                      ["T_env", None, T_env_dist]]

    parameters = un.Parameters(parameter_list)

To be able to use polynomial chaos methods when we have dependent parameters
we need to use the Rosenblatt transformation, which we enable by::

    UQ.quantify(rosenblatt=True)

The complete code example become:

.. literalinclude:: ../../../examples/coffee_dependent/uq_coffee_dependent_function.py
    :language: python
    :lines: 1-47