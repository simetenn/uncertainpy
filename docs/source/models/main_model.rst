.. _model:

Model
=====

Generally, models are created through the :py:class:`~uncertainpy.model.Model` class.
``Model`` takes the argument ``run`` and the optional arguments
``postprocess``, ``adaptive`` and ``labels``. ::

    model = un.Model(run=example_model,
                     postprocess=example_postprocess,
                     adaptive=True,
                     labels=["xlabel", "ylabel"])

The ``run`` argument must be a Python function that runs a
simulation on a specific model for a given set of model parameters,
and returns the simulation output.
We call such a function for a model function.
The ``postprocess`` argument is a Python function used to postprocess
the model output if required.
We go into details on the requirements of the ``postprocess`` and model
functions below.
``adaptive`` specifies whether the model uses adaptive time steps or not.
For adaptive models,
Uncertainpy automatically interpolates the output to a regular form
(the same number of points for each model evaluation).
Finally, ``labels`` allows the user to specify a list of labels to be
used on the axes when plotting the results.


Defining a model function
-------------------------

As explained above, the ``run`` argument is a Python function that runs
a simulation on a specific model for a given set of model parameters,
and returns the simulation output.
An example outline of a model function is::

    def example_model(parameter_1, parameter_2):
        # An algorithm for the model, or a script that runs
        # an external model, using the given input parameters.

        # Returns the model output and model time
        # along with the optional info object.
        return time, values, info

Such a model function has the following requirements:

    1. **Input.**
       The model function takes a number of arguments which define the
       uncertain parameters of the model.

    2. **Run the model.**
       The model must then be run using the parameters given as arguments.

    3. **Output.**
       The model function must return at least two objects,
       the model time (or equivalent, if applicable) and model output.
       Additionally, any number of optional info objects can be returned.
       In Uncertainpy,
       we refer to the time object as ``time``,
       the model output object as ``values``,
       and the remaining objects as ``info``.

            1. **Time** (``time``).
               The ``time`` can be interpreted as the x-axis of the model.
               It is used when interpolating (see below),
               and when certain features are calculated.
               We can return ``None`` if the model has no time
               associated with it.

            2. **Model output** (``values``).
               The model output must either be regular, or it must be possible to
               interpolate or postprocess the output (see :ref:`Features <main_features>`)
               to a regular form.

            3. **Additional info** (``info``).
               Some of the methods provided by Uncertainpy,
               such as the later defined model postprocessing,
               feature preprocessing,
               and feature calculations,
               require additional information from the model (e.g., the time a
               neuron receives an external stimulus).
               We recommend to use a
               single dictionary as info object,
               with key-value pairs for the information,
               to make debugging easier.
               Uncertainpy always uses a single dictionary as the
               ``info`` object.
               Certain features require that specific keys are present in this
               dictionary.

The model itself does not need to be implemented in Python.
Any simulator can be used,
as long as we can control the model parameters and retrieve the simulation
output via Python.
We can as a shortcut pass a model function to the
``model`` argument in :ref:`UncertaintyQuantification <UncertaintyQuantification>`,
instead of first having to create a ``Model`` instance.



Defining a postprocess function
-------------------------------

The ``postprocess`` function is used to postprocess the model output
before it is used in the uncertainty quantification.
Postprocessing does not change the model output sent to the feature
calculations.
This is useful if we need to transform the model output
This is useful if we need to transform the model output to a regular result
for the uncertainty quantification,
but still need to preserve the original model output to reliably
detect the model features.


.. image:: ../../images/diagram.png


This figure illustrates how the objects returned by the model
function are sent to both model ``postprocess``,
and feature ``preprocess`` (see :ref:`Features <main_features>`).
Functions associated with the model are in red while functions
associated with features are in green.


An example outline of the ``postprocess`` function is::

    def example_postprocess(time, values, info):
        # Postprocess the result to a regular form using time,
        # values, and info returned by the model function.

        # Return the postprocessed model output and time.
        return time_postprocessed, values_postprocessed


The only time postprocessing is required for Uncertainpy to work,
is when the model produces output that can not be interpolated to a regular
form by Uncertainpy.
Postprocessing is for example required for network models that give output in
the form of spike trains, i.e. time values indicating when a given neuron fires.
It should be noted that postprocessing of spike trains is already implemented
in Uncertainpy, in the :ref:`NestModel <nest_model>`.
For most purposes user defined postprocessing will not be necessary.

The requirements for the ``postprocess`` function are:

    1. **Input.**
       ``postprocess`` must take the objects returned by the
       model function as input arguments.

    2. **Postprocessing.**
       The model time (``time``) and output (``values``) must
       be postprocessed to a regular form, or to a form that can be
       interpolated to a regular form by Uncertainpy.
       If additional information is needed from the model, it can be passed
       along in the ``info`` object.

    3. **Output.**
       The ``postprocess`` function must return two objects:

            1. **Model time** (``time_postprocessed``).
               The first object is the postprocessed time (or equivalent)
               of the model.
               We can return ``None`` if the model has no time.
               Note that the automatic interpolation of the postprocessed
               time can only be performed if a postprocessed time is returned
               (if an interpolation is required).

            2. **Model output** (``values_postprocessed``).
               The second object is the postprocessed model output.


API Reference
-------------

.. autoclass:: uncertainpy.models.Model
   :members:
   :inherited-members:
   :undoc-members:


