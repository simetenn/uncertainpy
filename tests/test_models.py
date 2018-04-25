import os
import unittest

import numpy as np
from xvfbwrapper import Xvfb
# import nest

from uncertainpy.models import Model, NeuronModel, NestModel
from uncertainpy.core import Parallel
from uncertainpy.core import RunModel

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
        model = Model(run=model_function)

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
            Model(run=2)


    def test_set_parameters(self):
        parameters = {"a": -1, "b": -1}

        self.model.set_parameters(**parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)


    def test_init(self):
        def f(x):
            return x

        model = Model(run=f,
                      interpolate=True,
                      labels=["test x", "text y"])

        self.assertEqual(model.run, f)
        self.assertEqual(model.labels, ["test x", "text y"])
        self.assertTrue(model.interpolate)
        self.assertEqual(model.name, "f")
        self.assertEqual(model.suppress_graphics, False)

    def test_set_run(self):
        def f(x):
            return x

        model = Model()

        model.run = f
        self.assertEqual(model.run, f)
        self.assertEqual(model.name, "f")


    def test_validate_run(self):
        self.model.validate_run(("t", "U"))
        self.model.validate_run((1, 2, 3))
        self.model.validate_run([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            self.model.validate_run("123456")
            self.model.validate_run(np.linspace(0, 1, 100))

        with self.assertRaises(TypeError):
            self.model.validate_run(1)


    def test_validate_postprocess(self):
        self.model.validate_postprocess(("t", "U"))
        with self.assertRaises(ValueError):
            self.model.validate_postprocess("123456")
            self.model.validate_postprocess(np.linspace(0, 1, 100))
            self.model.validate_postprocess(1)
            self.model.validate_postprocess((1, 2, 3))

    def test_postprocess(self):
        result = self.model.postprocess(1, 2, 3)

        self.assertEqual(result, (1, 2))


    def test_assign_postprocess(self):
        model = Model(run=model_function)

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

        model = Model(run=model_function)
        with self.assertRaises(TypeError):
            model.postprocess = 12


    def test_postprocess_argument(self):
        def postprocess(time, values):
            return "time", "values"

        model = Model(run=model_function,
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

        parameters = {"kappa": 0.05, "T_env": 20}
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
        path = "models/interneuron_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        model = NeuronModel(file=file,
                            path=os.path.join(filedir, path),
                            name="interneuron",
                            stimulus_end=1000,
                            stimulus_start=1900,
                            test=12)

        self.assertEqual(model.info["test"], 12)
        self.assertEqual(model.info["stimulus_end"], 1000)
        self.assertEqual(model.info["stimulus_start"], 1900)
        self.assertEqual(model.name, "interneuron")
        self.assertEqual(model.suppress_graphics, True)

    def test_run(self):
        file = "mosinit.hoc"
        path = "models/interneuron_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        model = NeuronModel(file=file,
                            path=os.path.join(filedir, path))

        model_parameters = {"cap": 1.1, "Rm": 22000}

        # TODO uncomment for final version, works on travis
        # with Xvfb() as xvfb:
        #     model.run(**model_parameters)


    def test_assign_run(self):
        def test_run(a, b):
            return "time", "values"

        model = NeuronModel(run=test_run)

        model_parameters = {"a": 1.1, "b": 22000}
        time, values = model.run(**model_parameters)

        self.assertEqual(time, "time")
        self.assertEqual(values, "values")



    def test_parallel(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "models/interneuron_modelDB/")

        model = NeuronModel(path=path,
                            interpolate=True)

        parallel = Parallel(model=model)
        model_parameters = {"cap": 1.1, "Rm": 22000}

        with Xvfb() as xvfb:
            result = parallel.run(model_parameters)

        self.assertIn("NeuronModel", result.keys())

        self.assertIn("time", result["NeuronModel"].keys())
        self.assertIn("values", result["NeuronModel"].keys())
        self.assertIn("interpolation", result["NeuronModel"].keys())

    def test_runmodel(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "models/interneuron_modelDB/")

        model = NeuronModel(path=path,
                            interpolate=True)

        parameter_list = [["a", 1, None],
                          ["b", 2, None]]

        runmodel = RunModel(model=model, parameters=parameter_list, CPUs=1)
        uncertain_parameters = ["cap", "Rm"]
        nodes = np.array([[1.0, 1.1, 1.2], [21900, 22000, 22100]])

        result = runmodel.run(nodes, uncertain_parameters)

        self.assertEqual(result.model_name, "NeuronModel")

        self.assertIsInstance(result["NeuronModel"].evaluations[0], np.ndarray)
        self.assertIsInstance(result["NeuronModel"].evaluations[1], np.ndarray)
        self.assertIsInstance(result["NeuronModel"].evaluations[2], np.ndarray)

        self.assertEqual(len(result["NeuronModel"].evaluations[0]),
                         len(result["NeuronModel"].evaluations[1]))
        self.assertEqual(len(result["NeuronModel"].evaluations[1]),
                         len(result["NeuronModel"].evaluations[2]))

        self.assertIsInstance(result["NeuronModel"].time, np.ndarray)


class TestNestModel(unittest.TestCase):
    def test_init(self):
        model = NestModel(brunel_network)
        self.assertEqual(model.run, brunel_network)
        self.assertEqual(model.name, "brunel_network")
        self.assertEqual(model.suppress_graphics, False)

    def test_set_run(self):
        def f(x):
            return x

        model = NestModel()

        model.run = f
        self.assertEqual(model.run, f)
        self.assertEqual(model.name, "f")


    def test_run(self):
        model = NestModel(brunel_network)

        time, values = model.run(eta=2, g=5, delay=1.5, J_E=0.1)
        correct_values = np.array([39.4,  54.9,  68.5,  80.8,  90.9])

        self.assertEqual(time, 100)
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
