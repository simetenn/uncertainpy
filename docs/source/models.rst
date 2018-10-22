.. _models:

Models
======

In order to perform the uncertainty quantification and sensitivity
analysis of a model,
Uncertainpy needs to set the parameters of the model,
run the model using those parameters,
and receive the model output.
The main class for models is :ref:`Model <model>`, which is used to create
custom models.
Uncertainpy has built-in support for NEURON and NEST models,
found in the :ref:`NeuronModel <neuron_model>`  and :ref:`NestModel <nest_model>` classes
respectively.
Uncertainpy also has support for multiple model outputs through the use of
additional features.
It should be noted that while Uncertainpy is tailored towards neuroscience,
it is not restricted to only neuroscience models.
Uncertainpy can be used on any model that meets the criteria in
this section.



.. toctree::
    :maxdepth: 1

    models/main_model
    models/neuron_model
    models/nest_model
    models/multiple_outputs


