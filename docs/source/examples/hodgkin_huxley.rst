.. _hodgkin_huxley:

The Hodgkin-Huxley model
========================

Here we examine the canonical Hodgkin-Huxley model
(`Hodgkin and Huxley, 1952`_).
An uncertainty analysis of this model has been performed previously
(`Valderrama et al., 2014`_),
and we here we repeat a part of that study using Uncertainpy.

.. _Valderrama et al., 2014: https://mathematical-neuroscience.springeropen.com/articles/10.1186/2190-8567-5-3
.. _Hodgkin and Huxley, 1952: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf

The here used version of the Hodgkin-Huxley model has 11 parameters:

=============================    =========   =====================================    ==============================================
Parameter                        Value       Unit                                     Meaning
=============================    =========   =====================================    ==============================================
:math:`V_{0}`                    -10         mV                                       Initial voltage
:math:`C_\mathrm{m}`             1           :math:`\text{F}/\text{cm}^2`             Membrane capacitance
:math:`\bar{g}_{\mathrm{Na}}`    120         :math:`\text{mS/cm}^2`                   Sodium (Na) conductance
:math:`\bar{g}_{\mathrm{K}}`     36          :math:`\text{mS/cm}^2`                   Potassium (K) conductance
:math:`\bar{g}_{\mathrm{I}}`     0.3         :math:`\text{mS/cm}^2`                   Leak current conductance
:math:`E_\mathrm{Na}`            112         mV                                       Sodium equilibrium potential
:math:`E_\mathrm{K}`             -12         mV                                       Potassium equilibrium potential
:math:`E_\mathrm{I}`             10.613      mV                                       Leak current equilibrium potential
:math:`n_0`                      0.0011                                               Initial potassium activation gating variable
:math:`m_0`                      0.0003                                               Initial sodium activation gating variable
:math:`h_0`                      0.9998                                               Initial sodium inactivation gating variable
=============================    =========   =====================================    ==============================================


As in the previous study,
we assume each of these parameters have a uniform distribution in the range
:math:`\pm 10\%` around their original value.



We use uncertainty quantification and sensitivity analysis to explore how this
parameter uncertainty affect the model output,
i.e., the action potential response of the neural membrane potential :math:`V_m`
to an external current injection.
The model was exposed to a continuous external stimulus of :math:`140 \mu \mathrm{A/cm}^2`
starting at :math:`t = 0`,
and we examined the membrane potential in the time window between :math:`t` = 5 and 15 ms


As in the :ref:`cooling coffee cup example<coffee_cup>`,
we implement the Hodgkin-Huxley model as a Python function
(found in :download:`/examples/valderrama/valderrama.py <../../../examples/valderrama/valderrama.py>`):

.. literalinclude:: ../../../examples/valderrama/valderrama.py
    :language: python


We use this function when we perform the uncertainty quantification and
sensitivity analysis
(found in :download:`/examples/valderrama/uq_valderrama.py <../../../examples/valderrama/uq_valderrama.py>`).
We first initialize our model:

.. literalinclude:: ../../../examples/valderrama/uq_valderrama.py
    :language: python
    :lines: 6-8

Then we create the set of parameters:

.. literalinclude:: ../../../examples/valderrama/uq_valderrama.py
    :language: python
    :lines: 10-24

We use :py:meth:`~uncertainpy.Parameters.set_all_distributions` and
:py:func:`~uncertainpy.uniform` to give all parameters a uniform
distribution in the range :math:`\pm 10\%` around their fixed value.


.. literalinclude:: ../../../examples/valderrama/uq_valderrama.py
    :language: python
    :lines: 26-28

``set_all_distributions`` sets the distribution of all parameters.
If it receives a function as input,
it gives that function the fixed value of each parameter,
and expects to receive Chaospy functions.
``uniform`` is a closure.
It takes `interval` as input and returns a function which takes the
`fixed_value` of each parameter as input and returns a Chaospy distribution with this
`interval` around the `fixed_value`.
Ultimately the distribution of each parameter is set to `interval` around their
`fixed_value`::

    cp.Uniform(fixed_value - abs(interval/2.*fixed_value),
               fixed_value + abs(interval/2.*fixed_value)).



We can now use polynomial chaos expansions with point collocation to calculate the
uncertainty and sensitivity of the model:

.. literalinclude:: ../../../examples/valderrama/uq_valderrama.py
    :language: python
    :lines: 30-33

The complete code for the uncertainty quantification and sensitivity becomes:

.. literalinclude:: ../../../examples/valderrama/uq_valderrama.py
    :language: python