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
The function method is easiest so start with that approach.
We start by importing the packages we use::

    import uncertainpy as un
    import numpy as np                   # For the time array
    import chaospy as cp                 # To create distributions
    from scipy.integrate import odeint   # To integrate our equation

To create the model we define a Python function ``coffee_cup`` that
takes the uncertain parameters ``kappa`` and ``T_env`` as input arguments.
Inside this function we solve our equation by integrating it using
``scipy.integrate.odeint``, before we return the results.
The implementation of the model is::

    def coffee_cup(kappa, T_env):
        # Initial temperature and time
        time = np.linspace(0, 200, 150)
        T_0 = 95

        # The equation describing the model
        def f(T, time, kappa, T_env):
            return -kappa*(T - T_env)

        # Solving the equation by integration.
        values = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

        # Return time and model results
        return time, values

We could use this function directly in `UncertaintyQuantification`,
but we would like to have labels on the axes when plotting.
So we create a `Model` with the above run function and labels::

    # Create a model from coffee_cup function and add labels
    model = un.Model(run=coffee_cup,
                    labels=["Time [s]", "Temperature [C]"])

The next step is to define the uncertain parameters.
We use Chaospy to create the distributions.::

    # Create the distributions
    kappa_dist = cp.Uniform(0.025, 0.075)
    T_env_dist = cp.Uniform(15, 25)

    # Define a parameter list and use it to create the Parameters
    parameter_list = [["kappa", None, kappa_dist],
                      ["T_env", None, T_env_dist]]
    parameters = un.Parameters(parameter_list)


We can now calculate the uncertainty and sensitivity using polynomial chaos
expansions with point collocation,
which is the default option of ``quantify``::

    # Set up the uncertainty quantification
    uncertainty = un.UncertaintyQuantification(model=model,
                                               parameters=parameters)

    # Perform the uncertainty quantification using
    # polynomial chaos with point collocation (by default)
    uncertainty.quantify()

The complete code becomes:

.. literalinclude:: ../../../examples/coffee/uq_coffee_function.py
    :language: python


Using a class
-------------

The model can also be created as a class instead of using a function.
Most of the code is unchanged.
We create a class that inherits from ``uncertainpy.Model``, and to add the
labels we call on the constructor of the parent class ``uncertainpy.Model`` and
give it the labels.::

    class CoffeeCup(un.Model):
        # Add labels to the model
        def __init__(self):
            super(CoffeeCup, self).__init__(self,
                                            labels=["Time (s)", "Temperature (C)"])


Then we can implement the run method::

        # Define the run method
        def run(self, kappa, T_env):
            # Initial temperature and time
            T_0 = 95
            time = np.linspace(0, 200, 100)

            # The equation describing the model
            def f(T, time, kappa, T_env):
                return -kappa*(T - T_env)

            # Solving the equation by integration.
            temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

            # Return time and model results
            return time, temperature


Then instead of creating a model from a model function we just initialize our
``CoffeeCup`` model.::

    # Initialize the model
    model = CoffeeCup()

    # Create the distributions
    kappa_dist = cp.Uniform(0.025, 0.075)
    T_env_dist = cp.Uniform(15, 25)

    # Define a parameter list and use it to create the Parameters
    parameter_list = [["kappa", None, kappa_dist],
                    ["T_env", None, T_env_dist]]
    parameters = un.Parameters(parameter_list)

    # Set up the uncertainty quantification
    uncertainty = un.UncertaintyQuantification(model=model,
                                            parameters=parameters)

    # Perform the uncertainty quantification using
    # polynomial chaos with point collocation (by default)
    uncertainty.quantify()

