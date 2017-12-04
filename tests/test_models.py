import os
import unittest

import numpy as np
from xvfbwrapper import Xvfb

from uncertainpy.models import Model, NeuronModel, NestModel

from .models import HodgkinHuxley
from .models import CoffeeCup
from .models import izhikevich
from .models import brunel_network


from .testing_classes import TestingModel0d, TestingModel1d, TestingModel2d
from .testing_classes import TestingModelAdaptive, model_function


folder = os.path.dirname(os.path.realpath(__file__))


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model()


    def test_run_error(self):
        with self.assertRaises(NotImplementedError):
            self.model.run(a="parameter input")


    def test_run(self):
        model = Model(run_function=model_function)

        parameters = {"a": -1, "b": -1}
        time, values = model.run(**parameters)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) - 2))
        self.assertEqual(model.name, "model_function")

        model = Model()
        model.run = run_function=model_function

        parameters = {"a": -1, "b": -1}
        time, values = model.run(**parameters)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) - 2))
        self.assertEqual(model.name, "model_function")

        with self.assertRaises(TypeError):
            model.run = ""
            Model(run_function=2)


    def test_set_parameters(self):
        parameters = {"a": -1, "b": -1}

        self.model.set_parameters(**parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)


    def test_init(self):
        def f(x):
            return x

        model = Model(run_function=f,
                      adaptive=True,
                      labels=["test x", "text y"])

        self.assertEqual(model.run, f)
        self.assertEqual(model.labels, ["test x", "text y"])
        self.assertTrue(model.adaptive)
        self.assertEqual(model.name, "f")


    def test_set_run(self):
        def f(x):
            return x

        model = Model()

        model.run = f
        self.assertEqual(model.run, f)
        self.assertEqual(model.name, "f")


    def test_validate_run_result(self):
        self.model.validate_run_result(("t", "U"))
        self.model.validate_run_result((1, 2, 3))
        self.model.validate_run_result([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            self.model.validate_run_result("123456")
            self.model.validate_run_result(np.linspace(0,1,100))

        with self.assertRaises(TypeError):
            self.model.validate_run_result(1)


    def test_postprocess(self):
        result = self.model.postprocess(1, 2, 3)

        self.assertEqual(result, (1, 2))

class TestHodgkinHuxleyModel(unittest.TestCase):
    def test_run(self):
        model = HodgkinHuxley()

        parameters = {"gbar_Na": 120, "gbar_K": 36, "gbar_l": 0.3}
        model.run(**parameters)



class TestCoffeeCupModel(unittest.TestCase):
    def test_run(self):
        model = CoffeeCup()

        parameters = {"kappa": -0.05, "T_env": 20}
        model.run(**parameters)



class TestIzhikevichModel(unittest.TestCase):
    def test_run(self):
        model = izhikevich

        parameters = parameters = {"a": 0.02, "b": 0.2, "c": -50, "d": 2}
        model(**parameters)





class TestTestingModel0d(unittest.TestCase):
    def test_run(self):
        model = TestingModel0d()

        parameters = {"a": -1, "b": -1}
        time, values = model.run(**parameters)

        self.assertEqual(t, 1)
        self.assertEqual(U, -1)





class TestTestingModel1d(unittest.TestCase):
    def test_run(self):
        model = TestingModel1d()

        parameters = {"a": -1, "b": -1}
        time, values = model.run(**parameters)


        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) - 2))



class TestTestingModel2d(unittest.TestCase):
    def test_run(self):
        model = TestingModel2d()

        parameters = {"a": -1, "b": -2}
        time, values = model.run(**parameters)


        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U,
                                       np.array([np.arange(0, 10) -1,
                                                 np.arange(0, 10) -2])))




class TestTestingModelAdaptive(unittest.TestCase):
     def test_run(self):
        model = TestingModelAdaptive()

        parameters = {"a": 1, "b": 2}
        time, values = model.run(**parameters)


        self.assertTrue(np.array_equal(np.arange(0, 13), t))
        self.assertTrue(np.array_equal(np.arange(0, 13) + 3, U))




class TestNeuronModel(unittest.TestCase):
    def test_init(self):
        file = "mosinit.hoc"
        path = "models/dLGN_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        model = NeuronModel(file=file,
                            path=os.path.join(filedir, path))


    def test_run(self):
        file = "mosinit.hoc"
        path = "models/dLGN_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        model = NeuronModel(file=file,
                            path=os.path.join(filedir, path))

        model_parameters = {"cap": 1.1, "Rm": 22000}

        # TODO uncomment for final version, works on travis
        # with Xvfb() as xvfb:
        #     model.run(**model_parameters)



class TestNestModel(unittest.TestCase):
    def test_init(self):
        model = NestModel(brunel_network)
        self.assertEqual(model.run, brunel_network)
        self.assertEqual(model.name, "brunel_network")


    def test_set_run(self):
        def f(x):
            return x

        model = NestModel()

        model.run = f
        self.assertEqual(model.run, f)
        self.assertEqual(model.name, "f")


    def test_run(self):
        model = NestModel(brunel_network)

        time, values = model.run()

        correct_values = [5.6, 11.1, 15.2, 19.5, 22.4, 30.3, 36, 42.2,
                     47.1, 55.2, 60.8, 67.3, 76.8, 81.5, 88.3, 96.1]

        self.assertIsNone(t)
        self.assertEqual(U[0], correct_U)


    def test_postprocess(self):
        model = NestModel(brunel_network)

        time, values = model.run()
        correct_values = [5.6, 11.1, 15.2, 19.5, 22.4, 30.3, 36, 42.2,
                     47.1, 55.2, 60.8, 67.3, 76.8, 81.5, 88.3, 96.1]

        time, values = model.postprocess(t, correct_U)

        binary_spike = np.zeros(len(t))
        binary_spike[np.in1d(t, correct_U)] = 1

        self.assertTrue(np.array_equal(t, np.arange(0, 100.1, 0.1)))
        self.assertTrue(np.array_equal(U, binary_spike))


if __name__ == "__main__":
    unittest.main()
