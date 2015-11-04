# TODO Can remove the fitted parameter and test if the parameter has a distribution function?

from distribution import Distribution

class Parameters():
    def __init__(self, parameters, distributions, uncertain_parameters):
        """
        parameters: dict of all the default parameters the model has
        uncbertain_parameters: list of all parameters that shall be examined
        distribution_function: Either a function for all parameters, or a dictionary
                               with the distribution for each parameter
        """

        self.parameters = {}
        self.parameter_space = None
        self.distributions = distributions


        if type(uncertain_parameters) is str:
            self.uncertain_parameters = [uncertain_parameters]
        else:
            self.uncertain_parameters = uncertain_parameters


        if hasattr(distributions, '__call__'):
            for parameter in parameters:
                if parameter in uncertain_parameters:
                    self.parameters[parameter] = Parameter(parameter, parameters[parameter],
                                                           distributions, True)
                else:
                    self.parameters[parameter] = Parameter(parameter, parameters[parameter])
        else:
            for parameter in parameters:
                if parameter in uncertain_parameters:
                    self.parameters[parameter] = Parameter(parameter, parameters[parameter],
                                                           distributions[parameter], True)
                else:
                    self.parameters[parameter] = Parameter(parameter, parameters[parameter])


    def getUncertain(self, item):
        items = []
        for parameter in self.parameters.values():
            if parameter.uncertain:
                items.append(getattr(parameter, item))
        return items


    def get(self, item):
        if item in self.parameters.keys():
            return self.parameters[item]

        return [getattr(parameter, item) for parameter in self.parameters.values()]



class Parameter():
    def __init__(self, name, value, distribution_function=Distribution(0.1).normal,
                 uncertain=False):

        self.name = name
        self.value = value
        self.distribution_function = distribution_function
        self.uncertain = uncertain

        if self.uncertain:
            if hasattr(distribution_function, '__call__'):
                self.parameter_space = self.distribution_function(self.value)
            else:
                print "Distribution function is not a function"
                print "rewrite to use a general distribution instead of a function"
