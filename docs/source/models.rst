.. _models:

Models
======



In order to perform the uncertainty quantification and sensitivity
analysis of a model,
Uncertainpy needs to set the parameters of the model,
run the model using those parameters,
and receive the model output.
Uncertainpy has built-in support for NEURON and NEST models,
found in the \lstinline|NeuronModel|
(\cref{sec:neuron}) and \lstinline|NestModel| (\cref{sec:nest}) classes
respectively.
It should be noted that while Uncertainpy is tailored towards neuroscience,
it is not restricted to only neuroscience models.
Uncertainpy can be used on any model that meets the criteria in
this section.
Below, we first explain how to create custom models,
before explaining how to use \lstinline|NeuronModel| and
\lstinline|NestModel|.








.. automodule:: uncertainpy.models

.. toctree::
    :maxdepth: 1

    models/main_model
    models/neuron_model
    models/nest_model