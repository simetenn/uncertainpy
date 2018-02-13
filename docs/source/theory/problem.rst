.. _problem:

The problem definition
======================

Consider a model :math:`U` that depends on space :math:`\boldsymbol{x}` and time :math:`t`,
has :math:`D` uncertain input parameters :math:`\boldsymbol{Q} = \left[Q_1, Q_2, \ldots, Q_D \right]`,
and gives the output :math:`Y`:

.. math::

   Y = U(\boldsymbol{x}, t, \boldsymbol{Q}).

The output :math:`Y` can be any value within the output space :math:`\Omega_Y`
and has an unknown probability density function :math:`\rho_Y`.
The goal of an uncertainty quantification is to describe the unknown :math:`\rho_Y`
through statistical metrics.
We are only interested in the input and output of the model,
and we ignore all details on how the model works.
The model :math:`U` is thus considered a black box,
and may represent any model, for example a spiking neuron model that returns a
voltage trace,
or a network model that return a spike train.

We assume the model includes uncertain parameters that can be described
by a multivariate probability density function :math:`\rho_{\boldsymbol{Q}}`.
Examples of parameters that can be uncertain in neuroscience are the
conductance of a single ion channel,
or the synaptic weight between two species of neurons in a network.
If the uncertain parameters are independent,
the multivariate probability density function :math:`\rho_{\boldsymbol{Q}}` can be given as
separate univariate probability density functions :math:`\rho_{Q_i}`,
one for each uncertain parameter :math:`Q_i`.
The joint multivariate probability density function for the independent
uncertain parameters is then:

.. math::

    \rho_{\boldsymbol{Q}} = \prod_{i=1}^D \rho_{Q_i}.

In cases where the uncertain input parameters are dependent,
the multivariate probability density function :math:`\rho_{\boldsymbol{Q}}` must be defined directly.
We assume the probability density functions are known, and are not here
concerned with how they are determined.
They may be the product of a series of measurements, a parameter estimation,
or educated guesses made by experts.