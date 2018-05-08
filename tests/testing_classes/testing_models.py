from uncertainpy import Model
import numpy as np



def model_function(a=1, b=2):
    time = np.arange(0, 10)
    values = np.arange(0, 10) + a + b

    return time, values



class TestingModel0d(Model):
    def __init__(self):
        super(TestingModel0d, self).__init__(labels=["x"], logger_level=None)


    def run(self, a=1, b=2):
        time = 1
        values = b

        return time, values



class TestingModel1d(Model):
    def __init__(self):
        super(TestingModel1d, self).__init__(labels=["x", "y"], logger_level=None)

    def run(self, a=1, b=2):

        time = np.arange(0, 10)
        values = np.arange(0, 10) + a + b

        return time, values




class TestingModel2d(Model):
    def __init__(self):
        super(TestingModel2d, self).__init__(labels=["x", "y", "z"], logger_level=None)


    def run(self, a=1, b=2):
        time = np.arange(0, 10)
        values = np.array([np.arange(0, 10) + a, np.arange(0, 10) + b])

        return time, values



class TestingModelAdaptive(Model):
    def __init__(self, ignore=False):
        super(TestingModelAdaptive, self).__init__(labels=["x", "y"], interpolate=True, ignore=ignore, logger_level=None)

    def run(self, a=1, b=2):

        time = np.arange(0, 10 + a + b)
        values = np.arange(0, 10 + a + b) + a + b

        return time, values



class TestingModelConstant(Model):
    def __init__(self):
        Model.__init__(self, labels=["x", "y"], logger_level=None)


    def run(self, a=1, b=2):

        time = np.arange(0, 10)
        values = np.arange(0, 10)

        return time, values



class TestingModelNoTime(Model):
    def __init__(self):
        Model.__init__(self, labels=["y"], logger_level=None)


    def run(self, a=1, b=2):

        values = np.arange(0, 10) + a + b

        return values



class TestingModelNoTimeU(Model):
    def __init__(self):
        super(TestingModelNoTimeU, self).__init__(labels=[], logger_level=None)


    def run(self, a=1, b=2):
        return



class TestingModelThree(Model):
    def __init__(self):
        super(TestingModelThree, self).__init__(labels=["x", "y"], logger_level=None)


    def run(self, a=1, b=2):
        return 1, 2, 3

class TestingModelIncomplete(Model):
    def __init__(self):
        super(TestingModelIncomplete, self).__init__(labels=["x", "y"], logger_level=None)


    def run(self, a=1, b=2):
        return [1, 2, 3], [a, None, b]



class PostprocessErrorNumpy(Model):
    def __init__(self):
        super(PostprocessErrorNumpy, self).__init__(labels=["x", "y"], logger_level=None)

    def run(self, a=1, b=2):

        time = np.arange(0, 10)
        values = np.arange(0, 10) + a + b

        return time, values

    def postprocess(self, time, values):
        return np.linspace(0, 10, 100)


class PostprocessErrorOne(Model):
    def __init__(self):
       super(PostprocessErrorOne, self).__init__(labels=["x", "y"], logger_level=None)

    def run(self, a=1, b=2):

        time = np.arange(0, 10)
        values = np.arange(0, 10) + a + b

        return time, values

    def postprocess(self, time, values):
        return 1


class PostprocessErrorValue(Model):
    def __init__(self):
        super(PostprocessErrorValue, self).__init__(labels=["x", "y"], logger_level=None)

    def run(self, a=1, b=2):

        time = np.arange(0, 10)
        values = np.arange(0, 10) + a + b

        return time, values

    def postprocess(self, time, values):
        return (1, 2, 3)