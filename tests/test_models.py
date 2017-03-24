import numpy as np
import os
import unittest
import chaospy as cp

from uncertainpy.models import Model, NeuronModel
from uncertainpy import Parameters

from models import HodgkinHuxleyModel
from models import CoffeeCupPointModel
from models import IzhikevichModel

from testing_classes import TestingModel0d, TestingModel1d, TestingModel2d
from testing_classes import TestingModelAdaptive, model_function


folder = os.path.dirname(os.path.realpath(__file__))


class TestModel(unittest.TestCase):
    def setUp(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        self.model = Model(parameterlist)


    def test_init(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)
        model = Model(parameters)

        self.assertIsInstance(model.parameters, Parameters)


    def test_initParameterList(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        model = Model(parameterlist)

        self.assertIsInstance(model.parameters, Parameters)



    def test_run(self):
        with self.assertRaises(NotImplementedError):
            self.model.run(a="parameter input")



    def test_setDistribution(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)
        self.model = Model(parameters)


        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        self.model.setDistribution("gbar_Na", distribution_function)

        self.assertIsInstance(self.model.parameters["gbar_Na"].distribution, cp.Dist)


    def test_setDistributionNone(self):
        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        with self.assertRaises(KeyError):
            self.model.setDistribution("not a parameter", distribution_function)



    def test_setAllDistributions(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)
        self.model = Model(parameters)

        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        self.model.setAllDistributions(distribution_function)

        self.assertIsInstance(self.model.parameters["gbar_Na"].distribution, cp.Dist)
        self.assertIsInstance(self.model.parameters["gbar_K"].distribution, cp.Dist)
        self.assertIsInstance(self.model.parameters["gbar_l"].distribution, cp.Dist)


    def test_model_function(self):
        self.model.run = model_function

        parameters = {"a": -1, "b": -1}
        t, U = self.model.run(**parameters)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) - 2))


class TestHodgkinHuxleyModel(unittest.TestCase):
    def setUp(self):
        self.parameterlist = [["a", 1, None], ["b", 2, None]]
        self.model = HodgkinHuxleyModel(self.parameterlist)


    def test_run(self):
        parameters = {"gbar_Na": 120, "gbar_K": 36, "gbar_l": 0.3}
        self.model.run(**parameters)



class TestCoffeeCupPointModel(unittest.TestCase):
    def setUp(self):
        self.parameterlist = [["a", 1, None], ["b", 2, None]]
        self.model = CoffeeCupPointModel(self.parameterlist)


    def test_run(self):
        parameters = {"kappa": -0.05, "u_env": 20}
        self.model.run(**parameters)



class TestIzhikevichModel(unittest.TestCase):
    def setUp(self):
        self.parameterlist = [["a", 1, None], ["b", 2, None]]
        self.model = IzhikevichModel(self.parameterlist)

    def test_run(self):
        parameters = parameters = {"a": 0.02, "b": 0.2, "c": -50, "d": 2}
        self.model.run(**parameters)





class TestTestingModel0d(unittest.TestCase):
    def setUp(self):
        self.parameterlist = [["a", 1, None], ["b", 2, None]]
        self.model = TestingModel0d(self.parameterlist)


    def test_run(self):
        parameters = {"a": -1, "b": -1}
        t, U = self.model.run(**parameters)

        self.assertEqual(t, 1)
        self.assertEqual(U, -1)





class TestTestingModel1d(unittest.TestCase):
    def setUp(self):
        self.parameterlist = [["a", 1, None], ["b", 2, None]]
        self.model = TestingModel1d(self.parameterlist)


    def test_run(self):
        parameters = {"a": -1, "b": -1}
        t, U = self.model.run(**parameters)


        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) - 2))



class TestTestingModel2d(unittest.TestCase):
    def setUp(self):
        self.parameterlist = [["a", 1, None], ["b", 2, None]]
        self.model = TestingModel2d(self.parameterlist)



    def test_run(self):
        parameters = {"a": -1, "b": -2}
        t, U = self.model.run(**parameters)


        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U,
                                       np.array([np.arange(0, 10) -1,
                                                 np.arange(0, 10) -2])))




class TestTestingModelAdaptive(unittest.TestCase):
    def setUp(self):
        self.model = TestingModelAdaptive()


    def test_run(self):
        parameters = {"a": 1, "b": 2}
        t, U = self.model.run(**parameters)


        self.assertTrue(np.array_equal(np.arange(0, 13), t))
        self.assertTrue(np.array_equal(np.arange(0, 13) + 3, U))




class TestNeuronModel(unittest.TestCase):
    def setUp(self):
        model_file = "mosinit.hoc"
        model_path = "models/dLGN_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        self.model = NeuronModel(model_file=model_file,
                                 model_path=os.path.join(filedir, model_path))


    # def test_run(self):
    #     model_parameters = {"cap": 1.1, "Rm": 22000}
    #
    #     with Xvfb() as xvfb:
    #         self.model.run(model_parameters)
    #



if __name__ == "__main__":
    unittest.main()
