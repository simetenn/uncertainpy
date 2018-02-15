.. _coffee_cup:

A cooling coffee cup model
==========================

Here we show an example (found in ``examples/coffee_cup``) where we examine the
changes in temperature of a cooling coffee cup that follows Newton’s law of
cooling:

.. math::

    \frac{dT(t)}{dt} = -\kappa(T(t) - T_{env})

This equation tells how the temperature :math:`T` of the coffee cup changes with
time :math:`t`, when it is in an environment with temperature :math:`T_{env}`.
:math:`\kappa` is a proportionality constant that is characteristic of the system
and regulates how fast the coffee cup radiates heat to the environment. For
simplicity we set the initial temperature to a fixed value,
:math:`T_0 = 95 ^\circ\text{C}`, and let :math:`\kappa` and :math:`T_{env}` be
uncertain parameters.
We give the uncertain parameters in the following
distributions:

.. math::

    \kappa &= \mathrm{Uniform}(0.025, 0.075),

    T_{env} &= \mathrm{Uniform}(15, 25).



Using a function
----------------

There are two approaches to creating the model, using a function or a class.
The function method is easiest so we start with that approach.
The complete for this example can be found in
:download:`examples/coffee/uq_coffee_function.py <../../../examples/coffee/uq_coffee_function.py>`.
We start by importing the packages we use:

.. literalinclude:: ../../../examples/coffee/uq_coffee_function.py
    :language: python
    :lines: 1-4

To create the model we define a Python function ``coffee_cup`` that
takes the uncertain parameters ``kappa`` and ``T_env`` as input arguments.
Inside this function we solve our equation by integrating it using
``scipy.integrate.odeint``, before we return the results.
The implementation of the model function is:

.. literalinclude:: ../../../examples/coffee/uq_coffee_function.py
    :language: python
    :lines: 7-21


We could use this function directly in ``UncertaintyQuantification``,
but we would like to have labels on the axes when plotting.
So we create a `Model` with the above run function and labels:

.. literalinclude:: ../../../examples/coffee/uq_coffee_function.py
    :language: python
    :lines: 24-25

The next step is to define the uncertain parameters.
We use Chaospy to create the distributions, and create a dictionary that we
pass to ``Parameters``:

.. literalinclude:: ../../../examples/coffee/uq_coffee_function.py
    :language: python
    :lines: 27-35


We can now calculate the uncertainty and sensitivity using polynomial chaos
expansions with point collocation,
which is the default option of ``quantify``.

.. literalinclude:: ../../../examples/coffee/uq_coffee_function.py
    :language: python
    :lines: 37-42

The complete code becomes:

.. literalinclude:: ../../../examples/coffee/uq_coffee_function.py
    :language: python


Using a class
-------------

The model can also be created as a class instead of using a function.
Most of the code is unchanged.
The complete for this example is in
:download:`examples/coffee/uq_coffee_class.py <../../../examples/coffee/uq_coffee_class.py>`.
We create a class that inherits from :ref:`Model <model>`. To add the
labels we call on the constructor of the parent class and
give it the labels.

.. literalinclude:: ../../../examples/coffee/uq_coffee_class.py
    :language: python
    :lines: 7-11

We can then implement the run method:

.. literalinclude:: ../../../examples/coffee/uq_coffee_class.py
    :language: python
    :lines: 14-28

Now, instead of creating a model from a model function, we initialize our
``CoffeeCup`` model:

.. literalinclude:: ../../../examples/coffee/uq_coffee_class.py
    :language: python
    :lines: 31-32

While the rest is unchanged:

.. literalinclude:: ../../../examples/coffee/uq_coffee_class.py
    :language: python
    :lines: 34-49
