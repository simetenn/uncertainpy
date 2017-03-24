from uncertainpy import Model
import numpy as np



def model_function(a=1, b=2):
    t = np.arange(0, 10)
    U = np.arange(0, 10) + a + b

    return t, U



class TestingModel0d(Model):
    def __init__(self, parameters):
        Model.__init__(self, parameters=parameters)

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):
        t = 1
        U = b

        return t, U



class TestingModel1d(Model):
    def __init__(self, parameters):
        Model.__init__(self, parameters=parameters)


        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):

        t = np.arange(0, 10)
        U = np.arange(0, 10) + a + b

        return t, U



class TestingModel2d(Model):
    def __init__(self, parameters):
        Model.__init__(self, parameters=parameters)

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):
        t = np.arange(0, 10)
        U = np.array([np.arange(0, 10) + a, np.arange(0, 10) + b])

        return t, U



class TestingModelAdaptive(Model):
    def __init__(self, parameters, adaptive_model=True):
        Model.__init__(self, parameters=parameters, adaptive_model=adaptive_model)

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):

        t = np.arange(0, 10 + a + b)
        U = np.arange(0, 10 + a + b) + a + b

        return t, U



class TestingModelConstant(Model):
    def __init__(self, parameters):
        Model.__init__(self, parameters=parameters)

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):

        t = np.arange(0, 10)
        U = np.arange(0, 10)

        return t, U



class TestingModelNoTime(Model):
    def __init__(self, parameters):
        Model.__init__(self, parameters=parameters)

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):

        U = np.arange(0, 10) + a + b

        return U


class TestingModelNoTimeU(Model):
    def __init__(self, parameters):
        Model.__init__(self, parameters=parameters)

        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):
        return



class TestingModelThree(Model):
    def __init__(self, parameters):
        Model.__init__(self, parameters=parameters)


        self.xlabel = "x"
        self.ylabel = "y"


    def run(self, a=1, b=2):
        return 1, 2, 3
