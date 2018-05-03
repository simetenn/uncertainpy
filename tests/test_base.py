import os
import shutil
import unittest
from logging import Logger

from uncertainpy.core.base import Base, ParameterBase
from uncertainpy import Model, Features, Parameters

from .testing_classes import TestingFeatures, model_function


class TestBase(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.filename = os.path.join(self.output_test_dir, "output")

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        base = Base(model=model_function,
                    features=model_function,
                    logger_level="warning",
                    logger_config_filename=self.filename)

        self.assertIsInstance(base.model, Model)
        self.assertEqual(base.model.run, model_function)

        self.assertIsInstance(base.features, Features)
        self.assertEqual(base.features.features_to_run, ["model_function"])

        self.assertIsInstance(base.logger, Logger)


    def test_feature(self):
        base = Base()

        base.features = TestingFeatures()
        self.assertIsInstance(base._features, TestingFeatures)
        self.assertIsInstance(base.features, TestingFeatures)

        base.features = None
        self.assertIsInstance(base._features, Features)
        self.assertIsInstance(base.features, Features)
        self.assertEqual(base.features.features_to_run, [])


    def test_set_model(self):
        base = Base()

        base.model = model_function

        self.assertIsInstance(base.model, Model)
        self.assertIsInstance(base._model, Model)
        self.assertEqual(base.model.run, model_function)
        self.assertEqual(base.model.name, "model_function")

        with self.assertRaises(TypeError):
            base.model = ["list"]



class TestParameterBase(unittest.TestCase):
    def test_init(self):
        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        base = ParameterBase(parameters=parameter_list,
                             model=model_function,
                             features=model_function,)

        self.assertIsInstance(base.model, Model)
        self.assertEqual(base.model.run, model_function)

        self.assertIsInstance(base.features, Features)
        self.assertEqual(base.features.features_to_run, ["model_function"])

        self.assertIsInstance(base.parameters, Parameters)
        self.assertEqual(base.parameters["a"].value, 1)
        self.assertEqual(base.parameters["b"].value, 2)


    def test_set_parameters(self):
        base = ParameterBase()

        parameter_list = [["a", 1, None],
                         ["b", 2, None]]

        base.parameters = parameter_list

        self.assertIsInstance(base.parameters, Parameters)
        self.assertEqual(base.parameters["a"].value, 1)
        self.assertEqual(base.parameters["b"].value, 2)

        parameter_list = [["a", 1, None],
                          ["b", 2, None]]


        base.parameters = Parameters(parameter_list)

        self.assertIsInstance(base.parameters, Parameters)
        self.assertEqual(base.parameters["a"].value, 1)
        self.assertEqual(base.parameters["b"].value, 2)