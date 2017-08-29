import re
import fileinput
import sys
# from builtins import dict
import collections

import chaospy as cp


__all__ = ["Parameters", "Parameter"]

class Parameter(object):
    def __init__(self, name, value, distribution=None):
        """
        Parameter object

        Parameters
        ----------
        name: str
            name of the parameter
        value: number
            the value of the parameter
        distribution: {None, Chaospy distribution, Function that returns a Chaospy distribution}, optional
            The distribution of the parameter, used if the parameter is uncertain.
            Defaults to None.

        """
        self.name = name
        self.value = value

        self._distribution = None

        self.distribution = distribution


    @property
    def distribution(self):
        """
        A Chaospy distribution or a function that returns a Chaospy distribution.
        """
        return self._distribution

    @distribution.setter
    def distribution(self, new_distribution):
        if new_distribution is None:
            self._distribution = None
        elif isinstance(new_distribution, cp.Dist):
            self._distribution = new_distribution
        elif hasattr(new_distribution, '__call__'):
            self._distribution = new_distribution(self.value)
            if not isinstance(self._distribution, cp.Dist):
                raise TypeError("Function new_distribution does not return a Chaospy distribution")
        else:
            raise TypeError("new_distribution is neither a function nor a Chaospy distribution")



    def set_parameter_file(self, filename, value):
        """
        Set parameters to given value in a parameter file.

        Search `filename` for occurrences of ``name = number``
        and replace ``number`` with `value`.

        Parameters
        ----------
        filename: str
            name of file
        value: float, int
            new value to set in parameter file

        """
        search_string = r"(\A|\b)(" + self.name + r")(\s*=\s*)((([+-]?\d+[.]?\d*)|([+-]?\d*[.]?\d+))([eE][+-]?\d+)*)($|\b)"
        pattern = re.compile(search_string)

        for line in fileinput.input(filename, inplace=True):
            sys.stdout.write(pattern.sub(r"\g<1>\g<2>\g<3>" + str(value), line))


    def reset_parameter_file(self, filename):
        """
        Set the parameter to the original value in the parameter file, `filename`.

        Parameters
        ----------
        filename: str
            name of file

        """
        self.set_parameter_file(filename, self.value)


    def __str__(self):
        """
        Return a readable string describing the parameter.

        Returns
        -------
        out: str

        A string containing ``name``, ``value``, and if a parameter is uncertain.

        """
        if self.distribution is None:
            uncertain = ""
        else:
            uncertain = " - Uncertain"

        return "{parameter}: {value}{uncertain}".format(parameter=self.name, value=self.value, uncertain=uncertain)








# TODO use collections.mutablemapping
class Parameters(collections.MutableMapping):
    def __init__(self, parameterlist=[]):
        """
        A collection of parameters.

        Parameters
        ----------

        parameterlist: {list of Parameter objects, list [[name, value, distribution],...]}
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

        try:
            for i in parameterlist:
                if isinstance(i, Parameter):
                    self.parameters[i.name] = i
                else:
                    self.parameters[i[0]] = Parameter(*i)
        except TypeError as error:
            msg = "parameters must be either list of Parameter objects or list [[name, value, distribution], ...]"
            if not error.args:
                error.args = ("",)
            error.args = error.args + (msg,)
            raise


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


    def __iter__(self):
        return iter(self.parameters.values())

    def __str__(self):
        """
        Return a readable string
        """
        result = ""
        for name in sorted(self.parameters.keys()):
            result += str(self.parameters[name]) + "\n"

        return result.strip()


    def __len__(self):
        return len(self.parameters)


    def __setitem__(self, name, parameter):
        if not isinstance(parameter, Parameter):
            raise ValueError("parameter must be an instance of Parameter")
        self.parameters[name] = parameter


    def __delitem__(self, name):
        del self.parameters[name]


    def set_distribution(self, parameter, distribution):
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
        self.parameters[parameter].distribution = distribution


    def set_all_distributions(self, distribution):
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
            self.parameters[parameter].distribution = distribution


    def get_from_uncertain(self, attribute="name"):
        """
Get an attribute from all uncertain parameters(parameters that have a distribution)

Parameters
----------
Required arguments

attribute: "name" | "value" | "distribution"
    The name of the attribute to be returned

Returns
-------
List of the attribute of all uncertain parameters
        """

        items = []
        for parameter in self.parameters.values():
            if parameter.distribution is not None:
                items.append(getattr(parameter, attribute))
        return items


    # TODO implement __getitem__
    def get(self, attribute="name", parameter_names=None):
        """
Get the attribute of all parameters in parameter_names

Parameters
----------
Required arguments

attribute: "name" | "value" | "distribution"
    The name of the attribute to be returned
parameter_names: None | list | str
    A list of all parameters of which attribute should be returned.
    If None, the attribute all parameters are returned.
    Default is None.
Returns
-------
List of the attribute of all uncertain parameters
        """

        if parameter_names is None:
            parameter_names = self.parameters.keys()

        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        return_parameters = []
        for parameter_name in parameter_names:
            return_parameters.append(self.parameters[parameter_name])

        return [getattr(parameter, attribute) for parameter in return_parameters]


    def set_parameters_file(self, filename, parameters):
        """
Set all parameter to the original value in the parameter file, filename.

Parameters
----------
Required arguments

filename: str
    name of file
        """
        for parameter in parameters:
            self.parameters[parameter].set_parameter_file(filename, parameters[parameter])


    def reset_parameter_file(self, filename):
        """
Set the parameter to the original value in the parameter file, filename.

Parameters
----------
Required arguments

filename: str
    name of file
        """
        for parameter in self.parameters:
            self.parameters[parameter].set_parameter_file(filename, self.parameters[parameter].value)
