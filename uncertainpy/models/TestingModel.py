from model import Model
import numpy as np

class TestingModel0d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

    def run(self):
        self.t = None
        self.U = 2


class TestingModel1d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

    def run(self):
        self.t = np.arange(0, 10)
        self.U = self.t + self.a + self.b


class TestingModel2d(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters=parameters)

        self.a = 1
        self.b = 2

    def run(self):
        self.t = np.arange(0, 10)
        self.U = np.array([self.t + self.a, self.t + self.b])
