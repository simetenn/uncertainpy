from uncertainpy import Model
import numpy as np

class TestingModel0d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        t = 1
        U = self.b

        return t, U


class TestingModel1d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        t = np.arange(0, 10)
        U = np.arange(0, 10) + self.a + self.b

        return t, U


class TestingModel2d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        t = np.arange(0, 10)
        U = np.array([np.arange(0, 10) + self.a, np.arange(0, 10) + self.b])

        return t, U


class TestingModelNewProcess(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters, new_process=True)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        t = np.arange(0, 10)
        U = np.arange(0, 10) + self.a + self.b

        return t, U

class TestingModelAdaptive(Model):
    def __init__(self, parameters=None, adaptive_model=True):
        Model.__init__(self, parameters=parameters, adaptive_model=adaptive_model)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        t = np.arange(0, 10 + self.a + self.b)
        U = np.arange(0, 10 + self.a + self.b) + self.a + self.b

        return t, U

class TestingModelConstant(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        t = np.arange(0, 10)
        U = np.arange(0, 10)

        return t, U



class TestingModelNoTime(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

        U = np.arange(0, 10) + self.a + self.b

        return U


class TestingModelNoTimeU(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, parameters):
        return

class TestingModelThree(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, parameters):
        return 1, 2, 3
