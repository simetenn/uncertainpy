import unittest
import chaospy as cp
from uncertainpy import Parameter, Parameters


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.parameter = Parameter("gbar_Na", 120)

    def test_initNone(self):
        parameter = Parameter("gbar_Na", 120)

        self.assertTrue(parameter.name, "gbar_Na")
        self.assertTrue(parameter.value, 120)

    def test_initFunction(self):
        def distribution_function(x):
            return x

        parameter = Parameter("gbar_Na", 120, distribution_function)

        self.assertTrue(parameter.name, "gbar_Na")
        self.assertTrue(parameter.value, 120)
        # self.assertTrue(parameter.parameter_space, 120)
        # self.assertTrue(parameter.distribution_function, distribution_function)

    def test_initChaospy(self):
        parameter = Parameter("gbar_Na", 120, cp.Uniform(110, 130))

        self.assertTrue(parameter.name, "gbar_Na")
        self.assertTrue(parameter.value, 120)
        # self.assertTrue(parameter.parameter_space, 120)
        self.assertTrue(isinstance(parameter.distribution_function, cp.Dist))

    def test_setDistributionFunctionNone(self):
        distribution_function = None
        self.parameter.setDistribution(distribution_function)

    def test_setDistributionFunctionFunction(self):
        def distribution_function(x):
            return x

        self.parameter.setDistribution(distribution_function)
        self.assertTrue(self.parameter.parameter_space, 120)
        self.assertTrue(self.parameter.distribution_function, distribution_function)

    def test_setDistributionFunctionFunctionInt(self):
        distribution_function = 1
        with self.assertRaises(TypeError):
            self.parameter.setDistribution(distribution_function)



class TestParameters(unittest.TestCase):
    def setUp(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]
        self.parameters = Parameters(parameterlist)


    def test_initListNone(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)


    def test_initListChaospy(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]

        parameters = Parameters(parameterlist)


    def test_initObject(self):
        parameterlist = [Parameter("gbar_Na", 120),
                         Parameter("gbar_K", 36),
                         Parameter("gbar_l", 10.3)]

        parameters = Parameters(parameterlist)


    # def test_setDistrution(self):
    #
    # def setAllDistributions(self):
