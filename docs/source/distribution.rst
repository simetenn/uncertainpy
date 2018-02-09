Distribution
============

Functions (that work as closures) used to set the distribution of a
parameter to an `interval` around their original value through for example
:py:meth:`~uncertainpy.Parameters.set_all_distributions`.
An example::

    # Define a parameter list
    parameter_list = [["parameter_1", -67],
                      ["parameter_2", 22]]

    # Create the parameters
    parameters = un.Parameters(parameter_list)

    # Set all parameters to have a uniform distribution
    # within a 5% interval around their fixed value
    parameters.set_all_distributions(un.uniform(0.05))





API Reference
-------------

.. autofunction:: uncertainpy.uniform

.. autofunction:: uncertainpy.normal
