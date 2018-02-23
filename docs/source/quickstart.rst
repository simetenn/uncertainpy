.. _quickstart:

Quickstart
==========
This section gives a brief overview of what you need to know to perform an
uncertainty quantification and sensitivity analysis with Uncertainpy.
It only gives the most basic way of getting started, many more options than
shown here are available.

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

The arguments are given as instances of their corresponding Uncertainpy classes
(:ref:`Models <models>`, :ref:`Parameters <parameters>`, and :ref:`Features <features>`).
These classes are briefly described below.
After the problem is defined, an uncertainty quantification and sensitivity
analysis can be performed using the ``UncertaintyQuantification.quantify`` method.
Among others, ``quantify`` takes the following arguments::

    data = UQ.quantify(
        method="pc"``"mc",
        pc_method="collocation"``"spectral",
        rosenblatt=False``True
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
Additionally, the results from the uncertainty quantification are returned in
``data``,
as a ``Data`` object (see :ref:`Data <data>`).


Models
------

The easiest way to create a model is to use a Python function.
We need a Python function that runs a simulation on a
specified model for a given set of model parameters,
and returns the simulation output.
An example outline of a model function is::

    def example_model(parameter_1, parameter_2):
        # An algorithm for the model, or a script that runs
        # an external model, using the given input parameters.

        # Returns the model output and model time
        # along with the optional info object.
        return time, values, info

Such a model function can be given as the `model` argument to the
``UncertaintyQuantification`` class.
Note that sometimes an features or the preprocessing requires that additional
info object is required to be returned from the model.

For more on models see :ref:`Models`.


Parameters
----------


The parameters of a model are defined by two properties,
they must have (i) a name and (ii) either a fixed value or a distribution.
It is important that the name of the parameter is the same as the name given
as the input argument in the model function.
A parameter is considered uncertain if it has a probability distribution,
and the distributions are given as Chaospy distributions.
64 different univariate distributions are defined in Chaospy.
For a list of available distributions and detailed instructions on how to create
probability distributions with Chaospy,
see Section 3.3 in the `Chaospy paper`_.

.. _Chaospy paper: https://www.sciencedirect.com/science/article/pii/S1877750315300119

`parameters` is a dictionary with the above information,
the names of the parameters are the keys,
and the fixed values or distributions of the parameters are the values.
As an example, if we have two parameters,
where the first is named ``name_1`` and has a uniform probability
distributions in the interval :math:`[8, 16]`, and the second is named
``name_2`` and has a fixed value 42, the list become::

    import chaospy as cp
    parameters = {"name_1": cp.Uniform(8, 16), "name_2": 42}

The `parameter` argument in ``UncertaintyQuantification`` is such a dictionary.


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

    def example_feature(time, values, info):
        # Calculate the feature using time, values and info.

        # Return the feature times and values.
        return time_feature, values_feature

The `features` argument to ``UncertaintyQuantification`` can
be given as a list of feature functions we want to examine.


For more on features see :ref:`Features`.