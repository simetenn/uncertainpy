import numpy as np
import os
import unittest

from uncertainpy.models import HodkinHuxleyModel, CoffeeCupPointModel
from uncertainpy.models import IzhikevichModel, Model, NeuronModel

# def run_model(model, parameters):
#     model.load()
#     model.setParameterValues(parameters)
#     t, U = model.run()
#     model.save()
#     model.save(1)
#     model.cmd()
#
#
# def test_models():
#     models = {"CoffeeCupPointModel": {"kappa": -0.05,
#                                       "u_env": 20},
#               "HodkinHuxleyModel": {"gbar_Na": 120,
#                                     "gbar_K": 36,
#                                     "gbar_l": 0.3}}
#
#     for model_name in models:
#         model = getattr("uncertainpy.models", model_name)
#
#         run_model(model, models[model_name])


# def test_CoffeeCupPointModel():
#     from uncertainpy.models import CoffeeCupPointModel
#
#     model = CoffeeCupPointModel()
#     model.load()
#
#     parameters = {"kappa": -0.05, "u_env": 20}
#     model.setParameterValues(parameters)
#
#     t, U = model.run()
#
#     # def distribution_function(x):
#     #     return x
#     #
#     # model.setAllDistributions(distribution_function)
#     # model.setDistribution("kappa", distribution_function)
#
#     model.save()
#     t = np.load(".tmp_t.npy")
#     U = np.load(".tmp_U.npy")
#     os.remove(".tmp_U.npy")
#     os.remove(".tmp_t.npy")
#
#     model.save(1)
#     U = np.load(".tmp_U_%s.npy" % 1)
#     t = np.load(".tmp_t_%s.npy" % 1)
#     os.remove(".tmp_U_%s.npy" % 1)
#     os.remove(".tmp_t_%s.npy" % 1)
#
#     model.cmd()


class TestHodkinHuxleyModel(unittest.TestCase):
    def setUp(self):
        self.model = HodkinHuxleyModel()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = {"gbar_Na": 120, "gbar_K": 36, "gbar_l": 0.3}
        self.model.setParameterValues(parameters)


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


    def test_cmd(self):
        self.model.cmd()


class TestCoffeeCupPointModel(unittest.TestCase):
    def setUp(self):
        self.model = CoffeeCupPointModel()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = parameters = {"kappa": -0.05, "u_env": 20}
        self.model.setParameterValues(parameters)


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


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")



class TestIzhikevichModel(unittest.TestCase):
    def setUp(self):
        self.model = IzhikevichModel()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = parameters = {"a": 0.02, "b": 0.2, "c": -50, "d": 2}
        self.model.setParameterValues(parameters)


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


    def test_cmd(self):
        self.assertIsInstance(self.model.cmd(), list)
        self.assertEqual(self.model.cmd()[0], "python")


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model()


    def test_load(self):
        self.model.load()


    def test_setParametervalues(self):
        parameters = parameters = {"random1": 30.5, "random2": 0.8}
        self.model.setParameterValues(parameters)


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
