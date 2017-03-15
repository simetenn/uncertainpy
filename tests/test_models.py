import numpy as np
import os
import unittest
import sys
import subprocess
import chaospy as cp

from xvfbwrapper import Xvfb

from uncertainpy.models import Model, NeuronModel
from uncertainpy import Parameters

from models import HodgkinHuxleyModel
from models import CoffeeCupPointModel
from models import IzhikevichModel
from testing_classes import TestingModel0d, TestingModel1d, TestingModel2d
from testing_classes import TestingModel0dNoTime, TestingModel1dNoTime
from testing_classes import TestingModel2dNoTime, TestingModelNoU, TestingModel1dAdaptive


folder = os.path.dirname(os.path.realpath(__file__))


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


    def test_initParameterList(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        model = Model(parameterlist)

        self.assertIsInstance(model.parameters, Parameters)


    def test_set_properties(self):
        cmds = {"a": 1, "b": 2}

        self.model.set_properties(cmds)

        self.assertEqual(self.model.a, 1)
        self.assertEqual(self.model.b, 2)

        self.assertIn("a", self.model.additional_cmds)
        self.assertIn("b", self.model.additional_cmds)


    def test_reset_properties(self):
        cmds = {"a": 1, "b": 2}

        self.model.set_properties(cmds)
        self.model.reset_properties()
        self.assertEqual(self.model.additional_cmds, [])

        with self.assertRaises(AttributeError):
            self.model.a

        with self.assertRaises(AttributeError):
            self.model.b


    def test_run(self):
        with self.assertRaises(NotImplementedError):
            self.model.run({})


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
        self.assertEqual(result[0], sys.executable)


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

        self.assertIsInstance(self.model.parameters["gbar_Na"].distribution, cp.Dist)
        self.assertIsInstance(self.model.parameters["gbar_K"].distribution, cp.Dist)
        self.assertIsInstance(self.model.parameters["gbar_l"].distribution, cp.Dist)


    def test_setAllDistributionsNone(self):
        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        with self.assertRaises(AttributeError):
            self.model.setAllDistributions(distribution_function)


class TestHodgkinHuxleyModel(unittest.TestCase):
    def setUp(self):
        self.model = HodgkinHuxleyModel()


    def test_run(self):
        parameters = {"gbar_Na": 120, "gbar_K": 36, "gbar_l": 0.3}
        self.model.run(parameters)


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("HodgkinHuxleyModel", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = HodgkinHuxleyModel()

        parameters = {"gbar_Na": 120, "gbar_K": 36, "gbar_l": 0.3}
        self.model.run(parameters)
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


    def test_run(self):
        parameters = {"kappa": -0.05, "u_env": 20}
        self.model.run(parameters)


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
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = CoffeeCupPointModel()

        parameters = {"kappa": -0.05, "u_env": 20}
        self.model.run(parameters)
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

    def test_run(self):
        parameters = parameters = {"a": 0.02, "b": 0.2, "c": -50, "d": 2}
        self.model.run(parameters)
        

    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("IzhikevichModel", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)



    def test_useCase(self):
        self.model = IzhikevichModel()

        parameters = parameters = {"a": 0.02, "b": 0.2, "c": -50, "d": 2}
        self.model.run(parameters)
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


    def test_run(self):
        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertEqual(self.model.t, 1)
        self.assertEqual(self.model.U, -1)



    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel0d", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)



    def test_useCase(self):
        self.model = TestingModel0d()

        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertEqual(self.model.t, 1)
        self.assertEqual(self.model.U, -1)


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


    def test_run(self):
        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) - 2))


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel1d", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = TestingModel1d()

        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) - 2))

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



    def test_run(self):
        parameters = {"a": -1, "b": -2}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -2)

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) -1,
                                                 np.arange(0, 10) -2])))

    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel2d", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = TestingModel2d()
        parameters = {"a": -1, "b": -2}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -2)

        self.assertTrue(np.array_equal(self.model.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) -1,
                                                 np.arange(0, 10) -2])))


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



class TestTestingModel0dNoTime(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel0dNoTime()


    def test_run(self):
        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertIsNone(self.model.t)
        self.assertEqual(self.model.U, -1)


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel0dNoTime", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)



    def test_useCase(self):
        self.model = TestingModel0dNoTime()
        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertIsNone(self.model.t)
        self.assertEqual(self.model.U, -1)

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


    def test_run(self):
        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) - 2))


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel1dNoTime", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = TestingModel1dNoTime()
        parameters = {"a": -1, "b": -1}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U, np.arange(0, 10) - 2))

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


    def test_run(self):
        parameters = {"a": -1, "b": -2}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -2)

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) -1,
                                                 np.arange(0, 10) -2])))

    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel2dNoTime", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = TestingModel2dNoTime()
        parameters = {"a": -1, "b": -2}
        self.model.run(parameters)

        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -2)

        self.assertIsNone(self.model.t)
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) -1,
                                                 np.arange(0, 10) -2])))

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


    def test_run(self):
        parameters = {"a": -1, "b": -2}
        self.model.run(parameters)

        self.assertIsNone(self.model.t)
        self.assertIsNone(self.model.U)


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModelNoU", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = TestingModelNoU()
        parameters = {"a": -1, "b": -2}
        self.model.run(parameters)

        self.assertIsNone(self.model.t)
        self.assertIsNone(self.model.U)

        with self.assertRaises(ValueError):
            self.model.save()

        with self.assertRaises(ValueError):
            self.model.save(1)


class TestTestingModel1dAdaptive(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel1dAdaptive()


    def test_run(self):
        parameters = {"a": 1, "b": -2}
        self.model.run(parameters)

        t = np.arange(0, 13)
        U = np.arange(0, 13) + 3

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_cmd(self):
        result = self.model.cmd()

        self.assertIn("TestingModel1dAdaptive", result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], sys.executable)


    def test_useCase(self):
        self.model = TestingModel1dAdaptive()

        parameters = {"a": 1, "b": 2}
        self.model.run(parameters)

        t_result = np.arange(0, 13)
        U_result = np.arange(0, 13) + 3

        self.assertTrue(np.array_equal(self.model.t, t_result))
        self.assertTrue(np.array_equal(self.model.U, U_result))

        self.model.save()

        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.array_equal(t, t_result))
        self.assertTrue(np.array_equal(U, U_result))

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(t, t_result))
        self.assertTrue(np.array_equal(U, U_result))



class TestNeuronModel(unittest.TestCase):
    def setUp(self):
        model_file = "mosinit.hoc"
        model_path = "models/dLGN_modelDB/"

        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        self.model = NeuronModel(model_file=model_file,
                                 model_path=os.path.join(filedir, model_path))


    def test_run(self):
        with Xvfb() as xvfb:
            cmd = self.model.cmd()
            cmd += ["--CPU", "1", "--save_path", ""]


            simulation = subprocess.Popen(cmd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          env=os.environ.copy())
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
