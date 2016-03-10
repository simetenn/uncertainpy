import numpy as np
import os
import unittest

from uncertainpy.models import HodkinHuxleyModel, CoffeeCupPointModel
from uncertainpy.models import IzhikevichModel, Model, NeuronModel
from uncertainpy.models import TestingModel0d, TestingModel1d, TestingModel2d


class TestHodkinHuxleyModel(unittest.TestCase):
    def setUp(self):
        self.model = HodkinHuxleyModel()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = {"gbar_Na": -1, "gbar_K": -1, "gbar_l": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.gbar_Na, -1)
        self.assertEqual(self.model.gbar_K, -1)
        self.assertEqual(self.model.gbar_l, -1)


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


    def test_saveProcess(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNone(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNoneProcess(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(self.model.t))
        self.assertTrue(np.array_equal(self.model.U, U))



    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


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

        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


class TestCoffeeCupPointModel(unittest.TestCase):
    def setUp(self):
        self.model = CoffeeCupPointModel()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = parameters = {"kappa": -1, "u_env": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.kappa, -1)
        self.assertEqual(self.model.u_env, -1)


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


    def test_saveProcess(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))

    def test_saveTNone(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNoneProcess(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


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

        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


class TestIzhikevichModel(unittest.TestCase):
    def setUp(self):
        self.model = IzhikevichModel()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = parameters = {"a": -1, "b": -1, "c": -1, "d": -1}
        self.model.setParameterValues(parameters)


        self.assertEqual(self.model.a, -1)
        self.assertEqual(self.model.b, -1)
        self.assertEqual(self.model.c, -1)
        self.assertEqual(self.model.d, -1)


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


    def test_saveProcess(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNone(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNoneProcess(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(self.model.t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")



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

            self.assertIsInstance(self.model.cmd(), list)
            self.assertEqual(self.model.cmd()[0], "python")


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = parameters = {"random1": -1, "random2": -1}
        self.model.setParameterValues(parameters)

        self.assertEqual(self.model.random1, -1)
        self.assertEqual(self.model.random2, -1)


    def test_run(self):
        with self.assertRaises(NotImplementedError):
            self.model.run()


    def test_save(self):
        with self.assertRaises(ValueError):
            self.model.save()


    def test_saveProcess(self):
        with self.assertRaises(ValueError):
            self.model.save()


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")



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

        self.assertIsNone(self.model.t)
        self.assertEqual(self.model.U, 2)


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


    def test_saveProcess(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNone(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))

    def test_saveTNoneProcess(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(self.model.t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


    def test_useCase(self):
        self.model = TestingModel0d()
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

        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


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


    def test_saveProcess(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNone(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))

    def test_saveTNoneProcess(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(self.model.t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


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

        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


class TestTestingModel2d(unittest.TestCase):
    def setUp(self):
        self.model = TestingModel2d()


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
        self.assertTrue(np.array_equal(self.model.U,
                                       np.array([np.arange(0, 10) + 1,
                                                 np.arange(0, 10) + 2])))


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


    def test_saveProcess(self):
        self.model.t = 1
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.array_equal(self.model.t, t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_saveTNone(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save()
        t = np.load(".tmp_t.npy")
        U = np.load(".tmp_U.npy")
        os.remove(".tmp_U.npy")
        os.remove(".tmp_t.npy")

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))

    def test_saveTNoneProcess(self):
        self.model.t = None
        self.model.U = np.linspace(0, 10, 100)

        self.model.save(1)
        U = np.load(".tmp_U_%s.npy" % 1)
        t = np.load(".tmp_t_%s.npy" % 1)
        os.remove(".tmp_U_%s.npy" % 1)
        os.remove(".tmp_t_%s.npy" % 1)

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(self.model.U, U))


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


    def test_useCase(self):
        self.model = TestingModel2d()
        self.model.load()

        parameters = {"a": 1, "b": 2}
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

        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")

# class TestNeuronModel(unittest.TestCase):
#     def setUp(self):
#         model_file = "INmodel.hoc"
#         model_path = "../models/neuron_models/dLGN_modelDB/"
#         self.model = NeuronModel(model_file=model_file, model_path=model_path)
#
#     def test_load(self):
#         self.model.load()
#
#
#     def test_setParametervalues(self):
#         parameters = parameters = {"cap": 1.1, "Rm": 22000, "Vrest": -63,
#                                    "Epas": -67, "gna": 0.09, "nash": -52.6,
#                                    "gkdr": 0.37, "kdrsh": -51.2, "gahp": 6.4e-5,
#                                    "gcat": 1.17e-5}
#         self.model.load()
#         self.model.setParameterValues(parameters)
#
#
#     def test_run(self):
#         self.model.load()
#         self.model.run()
#
#
#     def test_save(self):
#         self.model.U = np.linspace(0, 10, 100)
#         self.model.t = 1
#         self.model.save()
#         t = np.load(".tmp_t.npy")
#         U = np.load(".tmp_U.npy")
#         os.remove(".tmp_U.npy")
#         os.remove(".tmp_t.npy")
#
#         self.model.save(1)
#         U = np.load(".tmp_U_%s.npy" % 1)
#         t = np.load(".tmp_t_%s.npy" % 1)
#         os.remove(".tmp_U_%s.npy" % 1)
#         os.remove(".tmp_t_%s.npy" % 1)
#
#
#     def test_cmd(self):
#         self.model.cmd()


if __name__ == "__main__":
    unittest.main()
