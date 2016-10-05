import numpy as np
import os
import unittest
import chaospy as cp

from xvfbwrapper import Xvfb
import subprocess

from uncertainpy.models import HodkinHuxleyModel, CoffeeCupPointModel
from uncertainpy.models import IzhikevichModel, Model, NeuronModel
from uncertainpy.models import TestingModel0d, TestingModel1d, TestingModel2d
from uncertainpy.models import TestingModel0dNoTime, TestingModel1dNoTime
from uncertainpy.models import TestingModel2dNoTime, TestingModelNoU, TestingModel1dAdaptive
from uncertainpy import Parameters


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model()


    def test_init(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)
        model = Model(parameters)

        self.assertIsInstance(model.parameters, Parameters)


    def test_load(self):
        self.model.load()


    def test_addCmds(self):
        cmds = {"a": 1, "b": 2}

        self.model.addCmds(cmds)

        self.assertEqual(self.model.a, 1)
        self.assertEqual(self.model.b, 2)

        self.assertIn("a", self.model.additional_cmds)
        self.assertIn("b", self.model.additional_cmds)


    def test_setParametervalues(self):
        parameters = parameters = {"random1": -1, "random2": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.random1, -1)
        self.assertEqual(self.model.random2, -1)


    def test_run(self):
        with self.assertRaises(NotImplementedError):
            self.model.run()


    def test_save(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(self.model.t, 1))
        self.assertTrue(np.array_equal(self.model.U, np.linspace(0, 10, 100)))


    def test_saveProcess(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, 1))
        self.assertTrue(np.array_equal(self.model.U, np.linspace(0, 10, 100)))


    def test_saveNoT(self):
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(self.model.t))
        self.assertTrue(np.array_equal(self.model.U, np.linspace(0, 10, 100)))


    def test_saveProcessNoT(self):
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(self.model.t))
        self.assertTrue(np.array_equal(self.model.U, np.linspace(0, 10, 100)))


    def test_saveNoU(self):
        self.model.t = 1
        with self.assertRaises(ValueError):
            self.model.save(1)


    def test_saveProcessNoU(self):
        self.model.t = 1
        with self.assertRaises(ValueError):
            self.model.save(1)


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("Model", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_setDistribution(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)
        self.model = Model(parameters)

        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        self.model.setDistribution("gbar_Na", distribution_function)

        self.assertIsInstance(self.model.parameters["gbar_Na"].parameter_space, cp.Dist)


    def test_setDistributionNone(self):
        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        with self.assertRaises(AttributeError):
            self.model.setDistribution("gbar_Na", distribution_function)



    def test_setAllDistributions(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)
        self.model = Model(parameters)

        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        self.model.setAllDistributions(distribution_function)

        self.assertIsInstance(self.model.parameters["gbar_Na"].parameter_space, cp.Dist)
        self.assertIsInstance(self.model.parameters["gbar_K"].parameter_space, cp.Dist)
        self.assertIsInstance(self.model.parameters["gbar_l"].parameter_space, cp.Dist)


    def test_setAllDistributionsNone(self):
        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        with self.assertRaises(AttributeError):
            self.model.setAllDistributions(distribution_function)


class TestHodkinHuxleyModel(unittest.TestCase):
    def setUp(self):
        self.model = HodkinHuxleyModel()


    def test_load(self):
        self.model.load()


    def test_run(self):
        self.model.run()


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("HodkinHuxleyModel", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_useCase(self):
        self.model = HodkinHuxleyModel()
        self.model.load()

        parameters = {"gbar_Na": 120, "gbar_K": 36, "gbar_l": 0.3}
        self.model.setParameterValues(parameters)
        self.model.run()
        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


class TestCoffeeCupPointModel(unittest.TestCase):
    def setUp(self):
        self.model = CoffeeCupPointModel()


    def test_load(self):
        self.model.load()


    def test_run(self):
        self.model.run()


    def test_save(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("CoffeeCupPointModel", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_useCase(self):
        self.model = CoffeeCupPointModel()
        self.model.load()

        parameters = parameters = {"kappa": -0.05, "u_env": 20}
        self.model.setParameterValues(parameters)
        self.model.run()
        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


class TestIzhikevichModel(unittest.TestCase):
    def setUp(self):
        self.model = IzhikevichModel()


    def test_load(self):
        self.model.load()


    def test_run(self):
        self.model.run()


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("IzhikevichModel", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")



    def test_useCase(self):
        self.model = IzhikevichModel()
        self.model.load()

        parameters = parameters = {"a": 0.02, "b": 0.2, "c": -50, "d": 2}
        self.model.setParameterValues(parameters)
        self.model.run()
        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))



class TestTestingModel0d(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel0d()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = {"a": -1, "b": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)


    def test_run(self):
        self.model.run()

        self.assertEqual(self.model.t, 1)
        self.assertEqual(self.model.U, 2)


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel0d", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")



    def test_useCase(self):
        self.model = TestingModel0d()
        self.model.load()

        parameters = {"a": 1, "b": 2}
        self.model.setParameterValues(parameters)
        self.model.run()


        self.assertEqual(self.model.t, 1)
        self.assertEqual(self.model.U, 2)

        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertEqual(self.model.t, 1)
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertEqual(self.model.t, t)
        self.assertTrue(np.array_equal(self.model.U, U))


class TestTestingModel1d(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel1d()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = {"a": -1, "b": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)


    def test_run(self):
        self.model.run()

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) + 3))


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel1d", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_useCase(self):
        self.model = TestingModel1d()
        self.model.load()

        parameters = {"a": 1, "b": 2}
        self.model.setParameterValues(parameters)
        self.model.run()

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) + 3))

        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


class TestTestingModel2d(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel2d()


    def test_load(self):
        self.model.load()



    def test_run(self):
        self.model.run()

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) + 1,
                                                 np.arange(0, 10) + 2])))

    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel2d", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_useCase(self):
        self.model = TestingModel2d()
        self.model.load()

        parameters = {"a": 1, "b": 2}
        self.model.setParameterValues(parameters)
        self.model.run()
        self.model.save()

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) + 1,
                                                 np.arange(0, 10) + 2])))

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))



class TestTestingModel0dNoTime(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel0dNoTime()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = {"a": -1, "b": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)


    def test_run(self):
        self.model.run()

        self.assertIsNone(self.model.t)
        self.assertEqual(self.model.U, 2)


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel0dNoTime", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")



    def test_useCase(self):
        self.model = TestingModel0dNoTime()
        self.model.load()

        parameters = {"a": 1, "b": 2}
        self.model.setParameterValues(parameters)
        self.model.run()


        self.assertIsNone(self.model.t)
        self.assertEqual(self.model.U, 2)

        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


class TestTestingModel1dNoTime(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel1dNoTime()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = {"a": -1, "b": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)


    def test_run(self):
        self.model.run()

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) + 3))


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel1dNoTime", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_useCase(self):
        self.model = TestingModel1dNoTime()
        self.model.load()

        parameters = {"a": 1, "b": 2}
        self.model.setParameterValues(parameters)
        self.model.run()

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) + 3))

        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)


        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


class TestTestingModel2dNoTime(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel2dNoTime()


    def test_load(self):
        self.model.load()


    def test_run(self):
        self.model.run()

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) + 1,
                                                 np.arange(0, 10) + 2])))

    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel2dNoTime", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_useCase(self):
        self.model = TestingModel2dNoTime()
        self.model.load()

        parameters = {"a": 1, "b": 2}
        self.model.setParameterValues(parameters)
        self.model.run()

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) + 1,
                                                 np.arange(0, 10) + 2])))
        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


class TestTestingModelNoU(unittest.TestCase):
    def setUp(self):
        self.model = TestingModelNoU()


    def test_load(self):
        self.model.load()


    def test_run(self):
        self.model.run()

        self.assertIsNone(self.model.t)
        self.assertIsNone(self.model.U)


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModelNoU", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "python")


    def test_useCase(self):
        self.model = TestingModelNoU()
        self.model.load()

        parameters = {"a": 1, "b": 2}
        self.model.setParameterValues(parameters)

        self.model.run()

        self.assertIsNone(self.model.t)
        self.assertIsNone(self.model.U)

        with self.assertRaises(ValueError):
            self.model.save()

        with self.assertRaises(ValueError):
            self.model.save(1)


    class TestTestingModel1dAdaptive(unittest.TestCase):
        def setUp(self):
            self.model = TestingModel1dAdaptive()


        def test_load(self):
            self.model.load()


        def test_run(self):
            self.model.run()

            t = np.arange(0, 13)
            U = np.arange(0, 13) + 3

            self.assertTrue(np.array_equal(self.model.t, t))
            self.assertTrue(np.array_equal(self.model.U, U))


        def test_cmd(self):
            result = self.model.cmd()

            self.assertIn("TestingModel1dAdaptive", result)
            self.assertIsInstance(result, list)
            self.assertEqual(result[0], "python")


        def test_useCase(self):
            self.model = TestingModelNoU()
            self.model.load()

            parameters = {"a": 1, "b": 2}
            self.model.setParameterValues(parameters)

            self.model.run()

            self.assertIsNone(self.model.t)
            self.assertIsNone(self.model.U)

            with self.assertRaises(ValueError):
                self.model.save()

            with self.assertRaises(ValueError):
                self.model.save(1)

class TestNeuronModel(unittest.TestCase):
    def setUp(self):
        model_file = "mosinit.hoc"
        model_path = "../models/neuron_models/dLGN_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)
        self.model = NeuronModel(model_file=model_file,
                                 model_path=os.path.join(filedir, model_path))

    def test_load(self):
        with Xvfb() as xvfb:
            self.model.load()


    def test_setParametervalues(self):
        parameters = parameters = {"cap": 1.1, "Rm": 22000, "Vrest": -63,
                                   "Epas": -67, "gna": 0.09, "nash": -52.6,
                                   "gkdr": 0.37, "kdrsh": -51.2, "gahp": 6.4e-5,
                                   "gcat": 1.17e-5}
        with Xvfb() as xvfb:
            self.model.load()
            self.model.setParameterValues(parameters)



    def test_run(self):
        with Xvfb() as xvfb:
            cmd = self.model.cmd()
            cmd += ["--CPU", "1", "--save_path", ""]

            simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ut, err = simulation.communicate()

        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)


    def test_save(self):
        with Xvfb() as xvfb:
            self.model.U = np.linspace(0, 10, 100)
            self.model.t = 1
            self.model.save()
            t = np.load(".tmp_t.npy")
            U = np.load(".tmp_U.npy")
            os.remove(".tmp_U.npy")
            os.remove(".tmp_t.npy")

            self.model.save(1)
            U = np.load(".tmp_U_%s.npy" % 1)
            t = np.load(".tmp_t_%s.npy" % 1)
            os.remove(".tmp_U_%s.npy" % 1)
            os.remove(".tmp_t_%s.npy" % 1)


    def test_cmd(self):
        self.model.cmd()




if __name__ == "__main__":
    unittest.main()
