import chaospy as cp

"""
Functions (that work as closures) used to set the distribution of a
parameter to an `interval` around their original value.
"""

def uniform(interval):
    """
    A closure that creates a function that takes a `parameter` as input and
    returns a uniform distribution with `interval` around `parameter`.

    Parameters
    ----------
    interval : int, float
        The interval of the uniform distribution around `parameter`.

    Returns
    -------
    distribution : function
        A function that takes `parameter` as input and returns a
        uniform distribution with `interval` around this `parameter`.


    Notes
    -----
    This function ultimately calculates:

    .. code-block:: Python

        cp.Uniform(parameter - abs(interval/2.*parameter),
                   parameter + abs(interval/2.*parameter)).
    """
    def distribution(parameter):
        if parameter == 0:
            raise ValueError("Creating a percentage distribution around 0 does not work")

        return cp.Uniform(parameter - abs(interval/2.*parameter),
                          parameter + abs(interval/2.*parameter))
    return distribution


def normal(interval):
    """
    A closure that creates a function that takes a `parameter` as input and
    returns a Gaussian distribution with standard deviation `interval*parameter`
    around `parameter`.

    Parameters
    ----------
    interval : int, float
        The interval of the standard deviation ``interval*parameter`` for the
        Gaussian distribution.

    Returns
    -------
    distribution : function
        A function that takes a `parameter` as input and
        returns a Gaussian distribution standard deviation ``interval*parameter``.

    Notes
    -----
    This function ultimately calculates:

    .. code-block:: Python

        cp.Normal(parameter, abs(interval*parameter))
    """
    def distribution(parameter):
        if parameter == 0:
            raise ValueError("Creating a percentage distribution around 0 does not work")

        return cp.Normal(parameter, abs(interval*parameter))
    return distribution
