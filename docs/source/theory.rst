.. _theory:

Theory
======

.. toctree::
    :maxdepth: 1
    :hidden:

    theory/problem
    theory/uq
    theory/sa
    theory/qmc
    theory/pce
    theory/rosenblatt



Here we give an overview of the theory behind uncertainty quantification and
sensitivity analysis with a focus on (quasi-)Monte Carlo methods and polynomial
chaos expansions, the methods implemented in Uncertainpy.

Uncertainty quantification and sensitivity analysis provide rigorous procedures
to analyse and characterize the effects of parameter uncertainty on the output
of a model.
The methods for uncertainty quantification and sensitivity analysis can be
divided into global and local methods.
Local methods keep all but one model parameter fixed and explore how much the
model output changes due to variations in that single parameter.
Global methods,
on the other hand, allows the entire parameter space to vary simultaneously.
Global methods can therefore identify complex dependencies between the model
parameters in terms of how they affect the model output.

The global methods can be further divided into intrusive and non-intrusive methods.
Intrusive methods require changes to the underlying model equations,
and are often challenging to implement.
Models in neuroscience are often created with the use of advanced simulators such
as `NEST`_ and `NEURON`_.
Modifying the underlying equations of models using these simulators is a
complicated task best avoided.
Non-intrusive methods, on the other hand, consider the model as a black box,
and can be applied to any model without needing to modify the model equations
or implementation.
Global, non-intrusive methods are therefore the methods of choice in Uncertainpy.
The uncertainty calculations in Uncertainpy is based on the Python package
`Chaospy`_,
which provides global non-intrusive methods for uncertainty quantification
and sensitivity analysis.

.. _NEST: http://www.nest-simulator.org/
.. _NEURON: https://www.neuron.yale.edu/neuron/
.. _Chaospy: https://github.com/jonathf/chaospy


We start by introducing the :ref:`problem definition <problem>`.
Next, we introduce the statistical measurements for :ref:`uncertainty quantification <uq>`
and :ref:`sensitivity analysis <sa>`.
Further, we give an introduction to :ref:`(quasi-)Monte Carlo methods <qmc>`
and :ref:`Polynomial chaos expansions <pce>`,
the two methods used to calculate the uncertainty and sensitivity in Uncertainpy.
We next explain how Uncertainpy handle cases with
:ref:`dependent model parameters <rosenblatt>`.
We note that insight into this theory is not a prerequisite for using
Uncertainpy.




* :ref:`Problem definition <problem>`
* :ref:`Uncertainty quantification <uq>`
* :ref:`Sensitivity analysis <sa>`
* :ref:`(Quasi-)Monte Carlo methods <qmc>`
* :ref:`Polynomial chaos expansions <pce>`
* :ref:`Dependency between uncertain parameters <rosenblatt>`

