from model import Model
import numpy as np

class TestingModel0d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.t = 1
        self.U = self.b


class TestingModel1d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.t = np.arange(0, 10)
        self.U = np.arange(0, 10) + self.a + self.b


class TestingModel1dAdaptive(Model):
    def __init__(self, parameters=None, adaptive_model=True):
        Model.__init__(self, parameters=parameters, adaptive_model=adaptive_model)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.t = np.arange(0, 10 + self.a + self.b)
        self.U = np.arange(0, 10 + self.a + self.b) + self.a + self.b


class TestingModel1dConstant(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.t = np.arange(0, 10)
        self.U = np.arange(0, 10)



class TestingModel2d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.t = np.arange(0, 10)
        self.U = np.array([np.arange(0, 10) + self.a, np.arange(0, 10) + self.b])



class TestingModel0dNoTime(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.U = self.b


class TestingModel1dNoTime(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.U = np.arange(0, 10) + self.a + self.b


class TestingModel2dNoTime(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"

    def run(self):
        self.U = np.array([np.arange(0, 10) + self.a, np.arange(0, 10) + self.b])


class TestingModelNoU(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self):
        pass
