import chaospy as cp

__all__ = ["Parameters", "Parameter"]
__version__ = "0.1"



class Parameter():
    def __init__(self, name, value, distribution=None):

        self.name = name
        self.value = value
        self.setDistribution(distribution)


    def setDistribution(self, distribution):
        if distribution is None:
            self.parameter_space = None
        elif isinstance(distribution, cp.Dist):
            self.parameter_space = distribution
        elif hasattr(distribution, '__call__'):
            self.parameter_space = distribution(self.value)
        else:
            raise TypeError("Argument is neither a function nor a Chaospy distribution")



class Parameters():
    def __init__(self, parameterlist):
        """
parameterlist = [[name1, value1, distribution1],
                 [name2, value2, distribution2],
                 ...]
or
parameterlist = [ParameterObject1, ParameterObject2,...]
        """
        self.parameters = {}

        for i in parameterlist:
            if isinstance(i, Parameter):
                self.parameters[i.name] = i
            else:
                self.parameters[i[0]] = Parameter(i[0], i[1], i[2])


    def setDistribution(self, parameter, distribution):
        self.parameters[parameter].setDistribution(distribution)


    def setAllDistributions(self, distribution):
        for parameter in self.parameters:
            self.parameters[parameter].setDistribution(distribution)


    def getUncertain(self, item="name"):
        items = []
        for parameter in self.parameters.values():
            if parameter.parameter_space is not None:
                items.append(getattr(parameter, item))
        return items


    def get(self, item="name"):
        if item in self.parameters.keys():
            return self.parameters[item]

        return [getattr(parameter, item) for parameter in self.parameters.values()]
