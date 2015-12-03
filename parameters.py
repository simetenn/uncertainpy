
class Parameters():
    def __init__(self, parameterlist):
        """
        parameterlist = [[name1, value1, distribution1], [name2, value2, distribution2],...]
        or
        parameterlist = [ParameterObject1, ParameterObject2,...]
        """

        self.parameters = {}
        #self.parameter_space = None



        for i in parameterlist:
            if type(i) == Parameter:
                self.parameters[i[0]] = i
            else:
                print i[2]
                self.parameters[i[0]] = Parameter(i[0], i[1], i[2])

    def setDistribution(self, parameter, distribution_function):
        self.parameters[parameter].setDistribution(distribution_function)

    def setAllDistributions(self, distribution_function):
        for parameter in self.parameters:
            self.parameters[parameter].setDistribution(distribution_function)


        # The first sets a given distribution to all parameters
        # if hasattr(distributions, '__call__'):
        #     for parameter in parameters:
        #         if parameter in uncertain_parameters:
        #             self.parameters[parameter] = Parameter(parameter, parameters[parameter],
        #                                                    distributions, True)
        #         else:
        #             self.parameters[parameter] = Parameter(parameter, parameters[parameter])
        # else:
        #     for parameter in parameters:
        #         if parameter in uncertain_parameters:
        #             self.parameters[parameter] = Parameter(parameter, parameters[parameter],
        #                                                    distributions[parameter], True)
        #         else:
        #             self.parameters[parameter] = Parameter(parameter, parameters[parameter])


    def getUncertain(self, item="name"):
        items = []
        for parameter in self.parameters.values():
            if parameter.distribution_function is not None:
                items.append(getattr(parameter, item))
        return items


    def get(self, item="name"):
        if item in self.parameters.keys():
            return self.parameters[item]

        return [getattr(parameter, item) for parameter in self.parameters.values()]



class Parameter():
    def __init__(self, name, value, distribution_function=None):

        self.name = name
        self.value = value
        self.setDistribution(distribution_function)


    def setDistribution(self, distribution_function):
        self.distribution_function = distribution_function

        if distribution_function is None:
            self.parameter_space = None
        else:
            if hasattr(distribution_function, '__call__'):
                self.parameter_space = self.distribution_function(self.value)
            else:
                raise TypeError("Distribution function is not a function")
