import chaospy as cp
import re
import fileinput
import sys

__all__ = ["Parameters", "Parameter"]
__version__ = "0.9"



class Parameter():
    def __init__(self, name, value, distribution=None):
        """
Parameter object

Parameters
----------
Required arguments

name: str
    name of the parameters
value: number
    the value of the paramter

Optional arguments

distribution: None | Chaospy distribution | Function that returns a Chaospy distribution
    The distribution of the parameter.
    Used if the parameter is uncertain.
    Default is None.
        """
        self.name = name
        self.value = value
        self.setDistribution(distribution)



    def setDistribution(self, distribution):
        """
Set the distribution of a parameter.

Parameters
----------
Required arguments

distribution: None | Chaospy distribution | Function that returns a Chaospy distribution
    The distribution of the parameter.
    A parameter is considered uncertain if if has a distributiona associated
    with it.
        """

        if distribution is None:
            self.distribution = None
        elif isinstance(distribution, cp.Dist):
            self.distribution = distribution
        elif hasattr(distribution, '__call__'):
            self.distribution = distribution(self.value)
            if not isinstance(self.distribution, cp.Dist):
                raise TypeError("Function does not return a Chaospy distribution")
        else:
            raise TypeError("Argument is neither a function nor a Chaospy distribution")



    def setParameterValue(self, filename, value):
        """
Set the parameter to given value in a parameter file.
Searches filename for occurences of 'name = #number' and replace the '#number' with value

Parameters
----------
Required arguments

filename: str
    name of file
value: number
    new value to set in parameter file
        """
        search_string = r"(\A|\b)(" + self.name + r")(\s*=\s*)((([+-]?\d+[.]?\d*)|([+-]?\d*[.]?\d+))([eE][+-]?\d+)*)($|\b)"
        pattern = re.compile(search_string)

        for line in fileinput.input(filename, inplace=True):
            sys.stdout.write(pattern.sub(r"\g<1>\g<2>\g<3>" + str(value), line))


    def resetParameterValue(self, filename):
        """
Set the parameter to the original value in the parameter file, filename.

Parameters
----------
Required arguments

filename: str
    name of file
    """
        self.setParameterValue(filename, self.value)


class Parameters():
    def __init__(self, parameterlist):
        """
A collection of parameters.

Parameters
----------
Required arguments

parameterlist: list of Parameter objects | list [[name, value, distribution],...]
    List the parameters that should be created.
    On the form:

    parameterlist = [[name1, value1, distribution1],
                     [name2, value2, distribution2],
                     ...]
        name: str
            Name of the parameter
        value: number
            Value of the parameter
        distribution: None | Chaospy distribution | Function that returns a Chaospy distribution
            The distribution of the parameter.
            A parameter is considered uncertain if if has a distributiona associated
            with it.

    or
    parameterlist = [ParameterObject1, ParameterObject2,...]
        """
        self.parameters = {}

        for i in parameterlist:
            if isinstance(i, Parameter):
                self.parameters[i.name] = i
            else:
                self.parameters[i[0]] = Parameter(i[0], i[1], i[2])


    def __getitem__(self, name):
        """
Return Parameter object with name

Parameters
----------
Required arguments

name: str
    name of parameter

Returns
-------
Parameter object
        """
        return self.parameters[name]


    def setDistribution(self, parameter, distribution):
        """
Set the distribution of a parameter.

Parameters
----------
Required arguments

parameter: str
    Name of parameter
distribution: None | Chaospy distribution | Function that returns a Chaospy distribution
    The distribution of the parameter.
    A parameter is considered uncertain if if has a distributiona associated
    with it.
        """
        self.parameters[parameter].setDistribution(distribution)


    def setAllDistributions(self, distribution):
        """
Set the distribution of all parameters.

Parameters
----------
Required arguments

distribution: None | Chaospy distribution | Function that returns a Chaospy distribution
    The distribution of the parameter.
    A parameter is considered uncertain if if has a distributiona associated
    with it.
        """
        for parameter in self.parameters:
            self.parameters[parameter].setDistribution(distribution)


    def getUncertain(self, prop="name"):
        """
Get a property of all uncertain parameters(parameters that have a distribution=

Parameters
----------
Required arguments

prop: "name" | "value" | "distribution"
    The name of the property to be returned

Returns
-------
List of the property of all uncertain parameters
        """

        items = []
        for parameter in self.parameters.values():
            if parameter.distribution is not None:
                items.append(getattr(parameter, prop))
        return items


    def get(self, prop="name"):
        """
Get a property of all uncertain parameters(parameters that have a distribution=

Parameters
----------
Required arguments

prop: "name" | "value" | "distribution"
    The name of the property to be returned

Returns
-------
List of the property of all uncertain parameters
        """

        if prop in self.parameters.keys():
            return self.parameters[prop]

        return [getattr(parameter, prop) for parameter in self.parameters.values()]


    def setParameterValues(self, filename, parameters):
        """
Set all parameter to the original value in the parameter file, filename.

Parameters
----------
Required arguments

filename: str
    name of file
        """
        for parameter in parameters:
            self.parameters[parameter].setParameterValue(filename, parameters[parameter])


    def resetParameterValues(self, filename):
        """
Set the parameter to the original value in the parameter file, filename.

Parameters
----------
Required arguments

filename: str
    name of file
        """
        for parameter in self.parameters:
            self.parameters[parameter].setParameterValue(filename, self.parameters[parameter].value)
