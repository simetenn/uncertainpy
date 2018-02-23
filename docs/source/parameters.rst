.. _parameters:

Parameters
==========

The parameters of a model are defined by two properties
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


The parameters are defined by the :ref:`Parameters <Parameters class>`  class.
``Parameters`` takes the argument `parameters`.
`parameters` can be on many different forms, but the most useful is
a dictionary with the above information,
the names of the parameters are the keys,
and the fixed values or distributions of the parameters are the values.
As an example, if we have two parameters,
where the first is named ``name_1`` and has a uniform probability
distributions in the interval :math:`[8, 16]`, and the second is named
``name_2`` and has a fixed value 42, the list become::

    import chaospy as cp
    parameters = {"name_1": cp.Uniform(8, 16), "name_2": 42}

And ``Parameters`` is initialized::

    parameters = un.Parameters(parameters=parameters)


The other possible forms that `parameters` can take are:

    * ``{name_1: parameter_object_1, name: parameter_object_2, ...}``
    * ``{name_1:  value_1 or Chaospy distribution, name_2:  value_2 or Chaospy distribution, ...}``
    * ``[parameter_object_1, parameter_object_2, ...]``,
    * ``[[name_1, value_1 or Chaospy distribution], ...]``.
    * ``[[name_1, value_1, Chaospy distribution or callable that returns a Chaospy distribution], ...]``

Where ``name`` is the name of the parameter and ``parameter_object`` is a ``Parameter``
object (see below).
The `parameter` argument in ``UncertaintyQuantification`` is either
``Parameters`` object, or a ``parameters`` dictionary/list as shown above.

Each parameter in ``Parameters`` is a :ref:`Parameter <Parameter>` object.
Each ``Parameter`` object is responsible for storing the name and fixed value
and/or distribution of each parameter.
It is initialized as::

    parameter = Parameter(name="name_1", distribution=cp.Uniform(8, 16))

In general you should not need to use ``Parameter``, it is mainly for internal
use in Uncertainpy

API Reference
-------------

.. toctree::
    :maxdepth: 1

    parameters/parameters
    parameters/parameter