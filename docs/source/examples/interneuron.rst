.. _interneuron:

A multi-compartment model of a thalamic interneuron
===================================================

In this example we illustrate how Uncertainpy can be used on models implemented
in `NEURON`_.
For this example, we select a previously published model of an interneuron in
the dorsal lateral geniculate nucleus `Halnes et al., 2011`_.
Since the model is in implemented in NEURON,
the original model can be used directly with Uncertainpy with the use of
:py:class:`~uncertainpy.models.NeuronModel`.
The code for this case study is found in
:download:`/examples/interneuron/uq_interneuron.py <../../../examples/interneuron/uq_interneuron.py>`.
To be able to run this example you require both the Neuron model,
as well as the interneuron model saved in a folder ``/interneuron_modelDB/``

.. _NEURON: https://www.neuron.yale.edu/neuron/
.. _Halnes et al., 2011: http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002160

In the original modeling study,
a set of 11 parameters were tuned manually through a series of trials and
errors until the interneuron model obtained the desired response characteristics.
The final parameter set is:

============================   ==========   ===================================     ================      =================================================================================================
Parameter                      Value        Unit                                    Neuron variable       Meaning
============================   ==========   ===================================     ================      =================================================================================================
:math:`E_\mathrm{pas}`         -67          mV                                      ``Epas``              Membrane reversal potential
:math:`R_\mathrm{m}`           22           :math:`\text{k}\Omega/\text{cm}^2`      ``Rm``                Membrane Resistance
:math:`g_{\mathrm{Na}}`        0.09         :math:`\text{S/cm}^2`                   ``gna``               Max :math:`\text{Na}^+`-conductance in soma
:math:`Sh_\mathrm{Na}`         -52.6        mV                                      ``nash``              Shift in :math:`\text{Na}^+`-kinetics
:math:`g_{\mathrm{Kdr}}`       0.37         :math:`\text{S/cm}^2`                   ``gkdr``              Max direct rectifying :math:`\text{K}^+`-conductance in soma
:math:`Sh_{\mathrm{Kdr}}`      -51.2        mV                                      ``kdrsh``             Shift in :math:`\text{K}_{dr}`-kinetics.
:math:`g_{\mathrm{CaT}}`       1.17e-5      :math:`\text{S/cm}^2`                   ``gcat``              Max T-type :math:`\text{Ca}^{2+}`-conductance in soma
:math:`g_{\mathrm{CaL}}`       9e-4         :math:`\text{S/cm}^2`                   ``gcal``              Max L-type :math:`\text{Ca}^{2+}`-conductance in soma
:math:`g_{\mathrm{h}}`         1.1e-4       :math:`\text{S/cm}^2`                   ``ghbar``             Max conductance of a non-specific hyperpolarization activated cation channel in soma
:math:`g_{\mathrm{AHP}}`       6.4e-5       :math:`\text{S/cm}^2`                   ``gahp``              Max afterhyperpolarizing :math:`\text{K}^+`-conductance in soma
:math:`g_{\mathrm{CAN}}`       2e-8         :math:`\text{S/cm}^2`                   ``gcanbar``           Max conductance of a :math:`\text{Ca}^{2+}`-activated non-specific cation channel in soma
============================   ==========   ===================================     ================      =================================================================================================


To perform an uncertainty quantification and sensitivity analysis of this model,
we assume each of these 11 parameters have a uniform uncertainty distribution
in the interval :math:`\pm 2.5\%` around their original value.
We create these parameters similar to what we did in the :ref:`Hodgkin-Huxley example <hodgkin_huxley>`:


.. literalinclude:: ../../../examples/interneuron/uq_interneuron.py
    :language: python
    :lines: 3-21


A point-to-point comparison of voltage traces is often uninformative,
and we therefore want to perform a feature based analysis of the model.
Since we examine a spiking neuron model,
we choose the features in :py:class:`~uncertainpy.features.SpikingFeatures`:

.. literalinclude:: ../../../examples/interneuron/uq_interneuron.py
    :language: python
    :lines: 23-24

We study the response of the interneuron to a somatic current injection
between :math:`1000 \text{ ms} < t < 1900 \text{ ms}`.
``SpikingFeatures`` needs to know the start and end time of this
stimulus to be able to calculate certain features.
They are specified through the ``stimulus_start`` and
``stimulus_end`` arguments when initializing ``NeuronModel``.
Additionally, the interneuron model uses adaptive time steps,
meaning we have to set ``adaptive=True``.
In this way we tell Uncertainpy to perform an interpolation to get the
output on a regular form before performing the analysis:
We also give the path to the folder where
the neuron model is stored with ``path="interneuron_modelDB/"``.
``NeuronModel`` loads the NEURON model from ``mosinit.hoc``,
sets the parameters of the model,
evaluates the model and returns the somatic membrane potential of the neuron.
``NeuronModel`` therefore does not require a model function.


.. literalinclude:: ../../../examples/interneuron/uq_interneuron.py
    :language: python
    :lines: 26-28

We set up the problem, adding our features before we use polynomial chaos
expansion with point collocation to compute the statistical metrics for
the model output and all features:

.. literalinclude:: ../../../examples/interneuron/uq_interneuron.py
    :language: python
    :lines: 30-34

The complete code becomes:

.. literalinclude:: ../../../examples/interneuron/uq_interneuron.py
    :language: python