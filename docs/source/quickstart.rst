Quickstart
==========
This section gives a brief overview of what you need to know to perform an
uncertainty quantification and sensitivity analysis with Uncertainpy.
An more thorough introduction can be found in the `Uncertainpy paper`_.

.. _Uncertainpy paper:

The uncertainty quantification and sensitivity analysis
includes three main components:

    * The **model** we want to examine.
    * The **parameters** of the model.
    * Specifications of **features** in the model output.

The model and the parameters are required,
while the feature specification is optional.
The above components are brought together in the
:ref:`UncertaintyQuantification` class.
This class is the main class to interact with,
and is a wrapper for the uncertainty calculations.

Uncertainty quantification
--------------------------

The :ref:`UncertaintyQuantification`  class is used to define the problem,
perform the uncertainty quantification, and to save and visualize the results.
Among others, ``UncertaintyQuantification`` takes the following arguments::

    UQ = un.UncertaintyQuantification(
            model=...,                       # Required
            parameters=...,                  # Required
            features=...,                    # Optional
        )

After the problem is defined, an uncertainty quantification and sensitivity
analysis can be performed using the ``UncertaintyQuantification.quantify`` method.
Among others, ``quantify`` takes the following arguments::

    UQ.quantify(
        method="pc"|"mc",
        pc_method="collocation"|"spectral",
        rosenblatt=False|True
    )

The `method` argument allows the user to choose whether Uncertainpy
should use polynomial chaos (``"pc"``) or quasi-Monte carlo (``"mc"``) methods to
calculate the relevant statistical metrics.
If polynomial chaos are chosen, `pc_method` further specifies whether point
collocation (``"collocation"``) or spectral projection
(``"spectral"``) methods is used to calculate the expansion
coefficients.
Finally, `rosenblatt` (``False`` or ``True``)
determines if the Rosenblatt transformation should be used.
The Rosenblatt is required if the uncertain parameters are dependent.
If nothing is specified,
Uncertainpy by default uses polynomial chaos based on point collocation without the
Rosenblatt transformation.
The results from the uncertainty quantification are automatically saved and
plotted.


Models
------

The easiest way to create a model is to use a Python function.
We need a Python function that runs a simulation on a
specified model for a given set of model parameters,
and returns the simulation output.
An example outline of a model function is::

    def example_model(parameter_1, parameter_2):
        # An algorithm for the model, or a script that runs the model,
        # with the given input parameters and returns the model output
        # and model time (time and values).

        return time, values

Such a model function can be given as the `model` argument to the
``UncertaintyQuantification`` class.
Note that sometimes an additional info object is required to be returned from
the model.

For more on models see :ref:`Models`.


Parameters
----------

The parameters of a model are defined by two properties;
their name and either a fixed value or a probability distribution.
A parameter is considered uncertain if it has a probability distribution.
The parameters are defined by the :ref:`Parameters class`.
The simplest way to create a set of parameters are to create a list on the form::

    parameter_list = [[parameter_name_1, value_1, distribution_1],
                      [parameter_name_2, value_2, distribution_2],
                       ...]

    parameters = un.Parameters(parameter_list=parameter_list)


The `parameter_name` is used to set the parameter in the model function.
This means the model function input arguments must have the same names as the
names given to the parameters.

`value` is the fixed value of the parameter,
and `distribution` is the distribution of the parameter.
``None`` can be given instead of a fixed value or a distribution if not applicable.
The ``distribution`` is given as Chaospy distributions.
For a list of available distributions and detailed instructions on how to create
probability distributions with Chaospy,
see Section 3.3 in the `Chaospy paper`_.

.. _Chaospy paper: https://www.sciencedirect.com/science/article/pii/S1877750315300119


We create two parameters uniformly distributed between :math:`[8, 16]` with the name
``parameter_1`` and ``parameter_1`` and no fixed value as::

    import chaospy as cp

    parameter_list = [["parameter_1", None, cp.Uniform(8, 16)],
                      ["parameter_2", None, cp.Uniform(8, 16)]]

    parameters = un.Parameters(parameter_list=parameter_list)

The `parameter` argument in `UncertaintyQuantification` is either
``Parameters`` object, or a ``parameter_list`` as shown above.


For more on parameters see :ref:`Parameters`.



Features
--------

Features are specific traits of the model output, and Uncertainpy has support
for performing uncertainty quantification and sensitivity analysis of features
of the model output,
in addition to the model output itself.
Features are defined by creating a Python function to calculate a specific
feature from the model output.
The feature function take the items returned by the model as as input arguments,
calculates a specific feature of this model output and returns the results.
quantification on.

The outline for a feature function is::

    def example_feature(time, values):
        # Calculate the feature using time and values,
        # then return the feature times and values

        return time_feature, values_feature

The `features` argument to ``UncertaintyQuantification`` can
be given as a list of feature functions we want to examine.


For more on features see :ref:`Features`.