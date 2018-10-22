.. _multiple_outputs:

Multiple model outputs
======================

Uncertainpy is usable with multiple model outputs.
However, it does unfortunately not have direct support for this,
you have to use a small trick.
Uncertainpy by default only performs an uncertainty quantification of the first
model output returned.
But you can return the additional model outputs in the
info dictionary,
and then define new features that extract each model output from the info
dictionary,
and then returns the additional model output.

Here is an example that shows how to do this::

    import uncertainpy as un
    import chaospy as cp

    # Example model with multiple outputs
    def example_model(parameter_1, parameter_2):
        # Perform all model calculations here

        time = ...

        model_output_1 = ...
        model_output_2 = ...
        model_output_3 = ...

        # We can store the additional model outputs in an info
        # dictionary
        info = {"model_output_2": model_output_2,
                "model_output_3": model_output_3}

        # Return time, model output and info dictionary
        # The first model output (model_output_1) is automatically used in the
        # uncertainty quantification
        return time, model_output_1, info


We can perform an uncertainty quantification of the other model outputs by
creating a feature for each of the additional model outputs by extracting the
output from the info dictionary and then return the output::

    def model_output_2(time, model_output_1, info):
        return time, info["model_output_2"]

    def model_output_3(time, model_output_1, info):
        return time, info["model_output_3"]

    feature_list = [model_output_2, model_output_3]

    # Define the parameter dictionary
    parameters = {"parameter_1": cp.Uniform(),
                "parameter_2": cp.Uniform()}

    # Set up the uncertainty quantification
    UQ = un.UncertaintyQuantification(model=example_model,
                                    parameters=parameters,
                                    features=feature_list)

    # Perform the uncertainty quantification using
    # polynomial chaos with point collocation (by default)
    data = UQ.quantify()


Alternatively, we can directly return all model outputs,
but you are then unable to use the built-in features in Uncertainpy::


    # Example model with multiple outputs
    def example_model(parameter_1, parameter_2):
        # Perform all model calculations here

        time = ...

        model_output_1 = ...
        model_output_2 = ...
        model_output_3 = ...

        # Return time, model output and info dictionary
        # The first model output (model_output_1) is automatically used in the
        # uncertainty quantification
    return time, model_output_1,  model_output_2, model_output_3


    # We can perform an uncertainty quantification of the other model
    # outputs by creating a feature for each of the additional
    # model outputs by extracting the output from the info dictionary and
    # then return the output

    def model_output_2(time, model_output_1, model_output_2, model_output_3):
        return time, model_output_2

    def model_output_3(time, model_output_1, model_output_2, model_output_3):
        return time, model_output_3
