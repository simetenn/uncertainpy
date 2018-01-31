import os
import unittest

import numpy as np
from xvfbwrapper import Xvfb
import nest

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

        self.assertTrue(np.array_equal(time, np.arange(0, 10)))
        self.assertTrue(np.array_equal(values, np.arange(0, 10) - 2))
        self.assertEqual(model.name, "model_function")

        model = Model()
        model.run = model_function

        parameters = {"a": -1, "b": -1}
        time, values = model.run(**parameters)

        self.assertTrue(np.array_equal(time, np.arange(0, 10)))
        self.assertTrue(np.array_equal(values, np.arange(0, 10) - 2))
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
            self.model.validate_run_result(np.linspace(0, 1, 100))

        with self.assertRaises(TypeError):
            self.model.validate_run_result(1)


    def test_postprocess(self):
        result = self.model.postprocess(1, 2, 3)

        self.assertEqual(result, (1, 2))


    def test_assign_postprocess(self):
        model = Model(run_function=model_function)

        parameters = {"a": -1, "b": -1}
        results = model.run(**parameters)

        time, values = model.postprocess(*results)

        self.assertTrue(np.array_equal(time, np.arange(0, 10)))
        self.assertTrue(np.array_equal(values, np.arange(0, 10) - 2))

        def postprocess(time, values):
            return "time", "values"

        model.postprocess = postprocess

        time, values = model.postprocess(*results)

        self.assertEqual(time, "time")
        self.assertEqual(values, "values")


    def test_postprocess_argument(self):
        def postprocess(time, values):
            return "time", "values"

        model = Model(run_function=model_function,
                      postprocess=postprocess)

        parameters = {"a": -1, "b": -1}
        results = model.run(**parameters)

        time, values = model.postprocess(*results)

        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

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

        self.assertEqual(time, 1)
        self.assertEqual(values, -1)





class TestTestingModel1d(unittest.TestCase):
    def test_run(self):
        model = TestingModel1d()

        parameters = {"a": -1, "b": -1}
        time, values = model.run(**parameters)


        self.assertTrue(np.array_equal(time, np.arange(0, 10)))
        self.assertTrue(np.array_equal(values, np.arange(0, 10) - 2))



class TestTestingModel2d(unittest.TestCase):
    def test_run(self):
        model = TestingModel2d()

        parameters = {"a": -1, "b": -2}
        time, values = model.run(**parameters)


        self.assertTrue(np.array_equal(time, np.arange(0, 10)))
        self.assertTrue(np.array_equal(values,
                                       np.array([np.arange(0, 10) -1,
                                                 np.arange(0, 10) -2])))




class TestTestingModelAdaptive(unittest.TestCase):
     def test_run(self):
        model = TestingModelAdaptive()

        parameters = {"a": 1, "b": 2}
        time, values = model.run(**parameters)


        self.assertTrue(np.array_equal(np.arange(0, 13), time))
        self.assertTrue(np.array_equal(np.arange(0, 13) + 3, values))




class TestNeuronModel(unittest.TestCase):
    def test_init(self):
        file = "mosinit.hoc"
        path = "models/dLGN_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        model = NeuronModel(file=file,
                            path=os.path.join(filedir, path),
                            name="lgn",
                            stimulus_end=1000,
                            stimulus_start=1900,
                            test=12)

        self.assertEqual(model.info["test"], 12)
        self.assertEqual(model.info["stimulus_end"], 1000)
        self.assertEqual(model.info["stimulus_start"], 1900)
        self.assertEqual(model.name, "lgn")


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

        correct_values = np.array([55.6, 71.4, 109.4, 129.4, 141.6, 159.1, 193.3, 213.9,
                                   233.6, 245.1, 278.3, 305.5, 318., 340.7, 361.1, 391.9,
                                   416.2, 457.2, 475.6, 512.4, 544.5, 584.7, 594.6, 608.8,
                                   637.2, 656.1, 694.6, 716.5, 732.1, 766.2, 795.8, 829.4,
                                   860., 876.4, 908.2, 928.2, 948.4, 986.7])

        self.assertEqual(time, 1000)
        self.assertTrue(np.allclose(values[0], correct_values, rtol=1e-5))


    def test_postprocess(self):
        model = NestModel(brunel_network)


        time, values = model.postprocess(4, [[0, 2, 3]])

        correct_time = np.arange(0, 4, 0.1)
        binary_spike =  np.zeros(len(time))
        binary_spike[0] = 1
        binary_spike[20] = 1
        binary_spike[30] = 1

        binary_spike = [binary_spike]
        self.assertTrue(np.array_equal(time, correct_time))
        self.assertTrue(np.array_equal(values, binary_spike))


if __name__ == "__main__":
    unittest.main()
