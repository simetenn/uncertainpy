class Parameters():
    def __init__(self, parameters, distributions, fitted_parameters):
        """
        parameters: dict of all the default parameters the model has
        fitted_parameters: list of all parameters that shall be examined
        distribution_functions: a function that
        """
        self.parameters = {}
        self.parameter_space = None
        self.dist = None

        if type(fitted_parameters) is str:
            self.fitted_parameter = [fitted_parameters]
        else:
            self.fitted_parameters = fitted_parameters


        if hasattr(distributions, '__call__'):
            for param in parameters:
                self.parameter[param] = Parameter(param, parameters[param],
                                                  distributions, param in fitted_parameters)
        else:
            for param in parameters:
                self.parameter[param] = Parameter(param, parameters[param],
                                                  distributions[param], param in fitted_parameters)



    def newParameterSpace(self):
        """
        Generalized parameter space creation
        """

        self.parameter_space = []
        for param in self.parameters:
            if self.parameters[param].fitted:
                self.parameter_space.append(self.parameters[param].parameter_space)
        return self.parameter_space


class Parameter():
    def __init__(self, name, value, distribution_function, fitted=True):

        self.name = name
        self.value = value
        self.distribution_function = distribution_function
        self.fitted = fitted

        if self.fitted:
            self.parameter_space = self.distribution_function(self.value)
