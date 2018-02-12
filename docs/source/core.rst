.. _core:

Core
====

This module contains the classes that are responsible for running the model and
calculate features of the model, both in parallel (:ref:`RunModel <run_model>` and
:ref:`Parallel <parallel>`), as well as the class for performing the
uncertainty calculations (:ref:`UncertaintyCalculations <uncertainty_calculations>`).
It also contains the base classes that are responsible for setting and updating
parameters, models and features across classes (:ref:`Base and ParameterBase <base>`).

.. toctree::
    :maxdepth: 1

    core/uncertainty_calculations
    core/base
    core/parallel
    core/run_model