import numpy as np
import os
import unittest

from uncertainpy.models import HodkinHuxleyModel, CoffeeCupPointModel


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
        self.model.load()
        self.model.run()
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
        self.model.load()
        self.model.run()
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
