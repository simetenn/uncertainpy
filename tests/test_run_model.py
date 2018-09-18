import unittest
import os
import shutil
import scipy.interpolate

import numpy as np

from uncertainpy import Parameters
from uncertainpy.core import RunModel
from uncertainpy.models import Model
from uncertainpy.features import Features, SpikingFeatures

from .testing_classes import TestingFeatures, model_function
from .testing_classes import TestingModel0d, TestingModel1d, TestingModel2d
from .testing_classes import TestingModelAdaptive




class TestRunModel(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.features = TestingFeatures(features_to_run=["feature0d",
                                                         "feature1d",
                                                         "feature2d",
                                                         "feature_invalid",
                                                         "feature_interpolate"])

        self.parameter_list = [["a", 1, None],
                               ["b", 2, None]]

        self.parameters = Parameters(self.parameter_list)

        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=self.features,
                                 logger_level="warning")


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        RunModel(model=TestingModel1d(),
                parameters=self.parameters,
                logger_level="error")


    def test_set_feature(self):
        self.runmodel.features = Features(logger_level="error")
        self.assertIsInstance(self.runmodel._features, Features)
        self.assertIsInstance(self.runmodel._parallel.features, Features)



    def test_set_model(self):
        self.runmodel = RunModel(TestingModel2d(),
                                 None,
                                 logger_level="error")

        self.runmodel.model = TestingModel1d()

        self.assertIsInstance(self.runmodel._model, TestingModel1d)
        self.assertIsInstance(self.runmodel._parallel.model, TestingModel1d)


    def test_set_model_none(self):
        self.runmodel = RunModel(TestingModel2d(),
                                 None,
                                 logger_level="error")
        self.runmodel.model = None

        self.assertIsNone(self.runmodel._model)
        self.assertIsNone(self.runmodel._parallel.model)


    def test_set_parameter_list(self):
        runmodel = RunModel(TestingModel2d(),
                            None,
                            logger_level="error")

        runmodel.parameters = self.parameter_list

        self.assertIsInstance(runmodel.parameters, Parameters)


    def test_set_parameter_error(self):
        runmodel = RunModel(TestingModel2d(),
                            None,
                            logger_level="error")

        with self.assertRaises(TypeError):
            runmodel.parameters = 2


    def test_set_model_function(self):
        self.runmodel = RunModel(TestingModel2d(),
                                 None,
                                 logger_level="error")

        self.runmodel.model = model_function

        self.assertIsInstance(self.runmodel.model, Model)
        self.assertIsInstance(self.runmodel._parallel.model, Model)


    def test_init_model_function(self):
        self.runmodel = RunModel(model_function,
                                 None,
                                 logger_level="error")

        self.assertIsInstance(self.runmodel.model, Model)
        self.assertIsInstance(self.runmodel._parallel.model, Model)


    def test_feature_function(self):
        def feature_function(time, values):
            return "time", "values"

        self.runmodel.features = feature_function
        self.assertIsInstance(self.runmodel.features, Features)

        time, values = self.runmodel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")


        self.assertEqual(self.runmodel.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(time, values):
            return "time", "values"

        def feature_function2(time, values):
            return "time2", "values2"


        self.runmodel.features = [feature_function, feature_function2]
        self.assertIsInstance(self.runmodel.features, Features)

        time, values = self.runmodel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")


        time, values = self.runmodel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        time, values = self.runmodel.features.feature_function2(None, None)
        self.assertEqual(time, "time2")
        self.assertEqual(values, "values2")


        time, values = self.runmodel._parallel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        time, values = self.runmodel._parallel.features.feature_function2(None, None)
        self.assertEqual(time, "time2")
        self.assertEqual(values, "values2")

        self.assertEqual(self.runmodel.features.features_to_run,
                         ["feature_function", "feature_function2"])


    def test_feature_functions_base(self):
        def feature_function(time, values):
            return "time", "values"

        def feature_function2(time, values):
            return "time2", "values2"

        implemented_features = ["nr_spikes", "time_before_first_spike",
                                "spike_rate", "average_AP_overshoot",
                                "average_AHP_depth", "average_AP_width",
                                "accommodation_index", "average_duration"]

        self.runmodel.features = SpikingFeatures([feature_function, feature_function2],
                                                 logger_level="error")
        self.assertIsInstance(self.runmodel.features, SpikingFeatures)

        time, values = self.runmodel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")


        time, values = self.runmodel.features.feature_function2(None, None)
        self.assertEqual(time, "time2")
        self.assertEqual(values, "values2")

        self.assertEqual(set(self.runmodel.features.features_to_run),
                         set(["feature_function", "feature_function2"] + implemented_features))


    def test_create_model_parameters(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        uncertain_parameters = ["a", "b"]

        result = self.runmodel.create_model_parameters(nodes, uncertain_parameters)

        self.assertEqual(result, [{"a": 0, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 3}])


    def test_create_model_parameters_one(self):
        nodes = np.array([0, 1, 2])
        uncertain_parameters = ["a"]

        result = self.runmodel.create_model_parameters(nodes, uncertain_parameters)

        self.assertEqual(result, [{"a": 0, "b": 2}, {"a": 1, "b": 2}, {"a": 2, "b": 2}])



    def test_evaluate_nodes_sequential_model_0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_invalid"])

        self.runmodel = RunModel(model=TestingModel0d(),
                                 parameters=self.parameters,
                                 features=features,
                                 CPUs=1,
                                 logger_level="error")

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])


        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel0d", "feature_invalid"]))


    def test_evaluate_nodes_parallel_model_0d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_invalid"])

        self.runmodel = RunModel(model=TestingModel0d(),
                                 parameters=self.parameters,
                                 features=features,
                                 CPUs=3,
                                 logger_level="error")


        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel0d", "feature_invalid"]))


    def test_evaluate_nodes_sequential_model_1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.CPUs = 1

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel1d", "feature_invalid", "feature_interpolate"]))


    def test_evaluate_nodes_parallel_model_1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.CPUs = 3

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel1d", "feature_invalid", "feature_interpolate"]))

    def test_evaluate_nodes_sequential_model_2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_invalid"])

        self.runmodel = RunModel(model=TestingModel2d(),
                                 parameters=self.parameters,
                                 features=features,
                                 CPUs=1,
                                 logger_level="error")

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel2d", "feature_invalid"]))


    def test_evaluate_nodes_parallel_model_2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_invalid"])

        self.runmodel = RunModel(model=TestingModel2d(),
                                 parameters=self.parameters,
                                 features=features,
                                 CPUs=3,
                                 logger_level="error")

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel2d", "feature_invalid"]))

    def test_evaluate_nodes_not_supress_graphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.model.suppress_graphics = False


        self.runmodel.evaluate_nodes(nodes, ["a", "b"])


    def test_evaluate_nodes_supress_graphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.model.suppress_graphics = True
        self.runmodel.evaluate_nodes(nodes, ["a", "b"])



    def test_results_to_data_model_1d_all_features(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=features,
                                 logger_level="error")

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        data = self.runmodel.results_to_data(results)

        features = list(data.keys())
        features.sort()
        self.assertEqual(features,
                         ["TestingModel1d", "feature0d", "feature1d", "feature2d"])


        self.assert_testingmodel1d(data)
        self.assert_feature_0d(data)
        self.assert_feature_1d(data)
        self.assert_feature_2d(data)





    def test_results_to_data_model_1d_feature_invalid(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature_invalid"])

        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=features,
                                 logger_level="error")

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])


        data = self.runmodel.results_to_data(results)

        self.assertIn("feature_invalid", data)

        self.assertTrue(np.isnan(data["feature_invalid"]["time"]))
        self.assertTrue(np.all(np.isnan(data["feature_invalid"].evaluations)))



    def test_results_to_data_model_1d_features_all_interpolate(self):

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_interpolate"],
                                   interpolate="feature_interpolate")

        self.runmodel = RunModel(model=TestingModelAdaptive(),
                                 parameters=self.parameters,
                                 features=features,
                                 logger_level="error")


        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        data = self.runmodel.results_to_data(results)

        features = list(data.keys())
        features.sort()

        self.assertEqual(features,
                         ["TestingModelAdaptive", "feature0d", "feature1d",
                          "feature2d", "feature_interpolate"])

        self.assertIn("TestingModelAdaptive", data)

        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["time"],
                                       np.arange(0, 15)))
        self.assertTrue(np.allclose(data["TestingModelAdaptive"].evaluations[0],
                                    np.arange(0, 15) + 1))
        self.assertTrue(np.allclose(data["TestingModelAdaptive"].evaluations[1],
                                    np.arange(0, 15) + 3))
        self.assertTrue(np.allclose(data["TestingModelAdaptive"].evaluations[2],
                                    np.arange(0, 15) + 5))


        self.assertTrue(np.array_equal(data["feature_interpolate"]["time"],
                                       np.arange(0, 15)))
        self.assertTrue(np.allclose(data["feature_interpolate"].evaluations[0],
                                    np.arange(0, 15) + 1))
        self.assertTrue(np.allclose(data["feature_interpolate"].evaluations[1],
                                    np.arange(0, 15) + 3))
        self.assertTrue(np.allclose(data["feature_interpolate"].evaluations[2],
                                    np.arange(0, 15) + 5))


        self.assert_feature_0d(data)
        self.assert_feature_1d(data)
        self.assert_feature_2d(data)


    def test_results_to_data_model_1d_interpolate_ignore(self):
        self.runmodel = RunModel(model=TestingModelAdaptive(ignore=True),
                                 parameters=self.parameters,
                                 logger_level="error")

        self.runmodel.model.interpolate = False
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        data = self.runmodel.results_to_data(results)

        self.assertEqual(list(data.keys()),
                         ["TestingModelAdaptive"])


        self.assertIn("TestingModelAdaptive", data)

        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["time"][0],
                                       np.arange(0, 11)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["time"][1],
                                       np.arange(0, 13)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["time"][2],
                                       np.arange(0, 15)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[0],
                                       np.arange(0, 11)) + 1)
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[1],
                                       np.arange(0, 13) + 3))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[2],
                                       np.arange(0, 15) + 5))


    def test_results_to_data_model_adaptive_no_interpolate_no_ignore(self):
        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_interpolate"],
                                   interpolate="feature_interpolate")

        self.runmodel = RunModel(model=TestingModelAdaptive(ignore=False),
                                 parameters=self.parameters,
                                 features=features,
                                 logger_level="error")

        self.runmodel.model.interpolate = False
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        results[0]["TestingModelAdaptive"]["values"] = [[], [1, 2]]


        data = self.runmodel.results_to_data(results)
        self.assertEqual(data.error, ["TestingModelAdaptive"])

        self.assertIn("TestingModelAdaptive", data)
        self.assertEqual(data["TestingModelAdaptive"].evaluations[0],
                        [[], [1, 2]])
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].time[0],
                                       np.arange(0, 11)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].time[1],
                                       np.arange(0, 13)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].time[2],
                                       np.arange(0, 15)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[0],
                                       np.arange(0, 11)) + 1)
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[1],
                                       np.arange(0, 13) + 3))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[2],
                                       np.arange(0, 15) + 5))


        self.assert_feature_0d(data)
        self.assert_feature_1d(data)
        self.assert_feature_2d(data)

        self.assertIn("feature_interpolate", data)

        self.assertTrue(np.array_equal(data["feature_interpolate"]["time"],
                                       np.arange(0, 15)))
        self.assertTrue(np.allclose(data["feature_interpolate"].evaluations[0],
                                    np.arange(0, 15) + 1))
        self.assertTrue(np.allclose(data["feature_interpolate"].evaluations[1],
                                    np.arange(0, 15) + 3))
        self.assertTrue(np.allclose(data["feature_interpolate"].evaluations[2],
                                    np.arange(0, 15) + 5))


    def test_results_to_data_model_1d_spiketrain_ignore(self):
        self.runmodel = RunModel(model=TestingModelAdaptive(ignore=True),
                                 parameters=self.parameters,
                                 logger_level="error")

        self.runmodel.model.interpolate = False
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        results[0]["TestingModelAdaptive"]["values"] = [[], [1, 2]]


        data = self.runmodel.results_to_data(results)

        self.assertEqual(list(data.keys()),
                         ["TestingModelAdaptive"])

        self.assertIn("TestingModelAdaptive", data)

        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].time[0],
                                       np.arange(0, 11)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].time[1],
                                       np.arange(0, 13)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].time[2],
                                       np.arange(0, 15)))

        self.assertEqual(data["TestingModelAdaptive"].evaluations[0],
                        [[], [1, 2]])
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[0],
                                       np.arange(0, 11)) + 1)
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[1],
                                       np.arange(0, 13) + 3))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[2],
                                       np.arange(0, 15) + 5))


    def test_results_to_data_model_1d_interpolate_ignore_and_interpolate(self):
        self.runmodel = RunModel(model=TestingModelAdaptive(ignore=True),
                                 parameters=self.parameters,
                                 logger_level="error")

        self.runmodel.model.interpolate = True
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])


        data = self.runmodel.results_to_data(results)

        self.assertEqual(list(data.keys()),
                         ["TestingModelAdaptive"])


        self.assertIn("TestingModelAdaptive", data)

        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["time"][0],
                                       np.arange(0, 11)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["time"][1],
                                       np.arange(0, 13)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["time"][2],
                                       np.arange(0, 15)))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[0],
                                       np.arange(0, 11)) + 1)
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[1],
                                       np.arange(0, 13) + 3))
        self.assertTrue(np.array_equal(data["TestingModelAdaptive"].evaluations[2],
                                       np.arange(0, 15) + 5))


    def test_results_to_data_feature_0d(self):

        features = TestingFeatures(features_to_run=["feature0d",],
                                   interpolate="feature0d")

        self.runmodel = RunModel(model=TestingModelAdaptive(),
                                 parameters=self.parameters,
                                 features=features,
                                 logger_level="error")

        results = [{"feature0d": {"values": 1, "time": np.nan}},
                   {"feature0d": {"values": 1, "time": np.nan}},
                   {"feature0d": {"values": 1, "time": np.nan}}]
        data = self.runmodel.results_to_data(results)

        self.assert_feature_0d(data)



    # def test_results_to_datainterpolateError(self):
    #     self.runmodel = RunModel(TestingModelAdaptive(interpolate=True),
    #                              features=TestingFeatures(),
    #                              suppress_model_output=True)
    #
    #
    #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    #
    #
    #     results = self.runmodel.evaluate_nodes(nodes)
    #     self.runmodel.results_to_data(results)
    #
    #     self.assertEqual(set(self.runmodel.data.U.keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "TestingModelAdaptive", "feature_invalid"]))
    #
    #     self.assertEqual(set(self.runmodel.data.t.keys()),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "TestingModelAdaptive", "feature_invalid"]))
    #
    #     self.assertEqual(set(self.runmodel.data.feature_list),
    #                      set(["feature0d", "feature1d", "feature2d",
    #                           "TestingModelAdaptive", "feature_invalid"]))
    #
    #     self.assertIn("TestingModelAdaptive", self.runmodel.data.U.keys())
    #     self.assertTrue(np.array_equal(self.runmodel.data.t["TestingModelAdaptive"],
    #                                    np.arange(0, 10)))
    #     self.assertTrue(np.allclose(self.runmodel.data.U["TestingModelAdaptive"][0],
    #                                 np.arange(0, 10) + 1))
    #     self.assertTrue(np.allclose(self.runmodel.data.U["TestingModelAdaptive"][1],
    #                                 np.arange(0, 10) + 3))
    #     self.assertTrue(np.allclose(self.runmodel.data.U["TestingModelAdaptive"][2],
    #                                 np.arange(0, 10) + 5))
    #
    #     self.assert_feature_0d(self.runmodel.data)
    #     self.assert_feature_1d(self.runmodel.data)
    #     self.assert_feature_2d(self.runmodel.data)
    #     self.assert_feature_invalid(self.runmodel.data)




    def assert_feature_0d(self, data):
        self.assertIn("feature0d", data)

        self.assertTrue(np.isnan(data["feature0d"]["time"]))
        self.assertTrue(np.array_equal(data["feature0d"].evaluations, [1, 1, 1]))
        self.assertEqual(data["feature0d"]["labels"], ["feature0d"])


    def assert_feature_1d(self, data):
        self.assertIn("feature1d", data)

        self.assertTrue(np.array_equal(data["feature1d"]["time"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature1d"].evaluations[0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature1d"].evaluations[1],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature1d"].evaluations[2],
                                       np.arange(0, 10)))
        self.assertEqual(data["feature1d"]["labels"], ["feature1d x", "feature1d y"])



    def assert_feature_2d(self, data):
        self.assertIn("feature2d", data)
        self.assertTrue(np.array_equal(data["feature2d"]["time"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature2d"].evaluations[0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(data["feature2d"].evaluations[1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(data["feature2d"].evaluations[2],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertEqual(data["feature2d"]["labels"], ["feature2d x",
                                                       "feature2d y",
                                                       "feature2d z"])


    def assert_testingmodel1d(self, data):
        self.assertIn("TestingModel1d", data)
        self.assertTrue(np.array_equal(data["TestingModel1d"]["time"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["TestingModel1d"].evaluations[0],
                                       np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(data["TestingModel1d"].evaluations[1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(data["TestingModel1d"].evaluations[2],
                                       np.arange(0, 10) + 5))
        self.assertEqual(data["TestingModel1d"]["labels"], ["x", "y"])


    # # def assert_feature_invalid(self, data):
    # #     self.assertIn("feature_invalid", data.U.keys())
    # #     self.assertTrue(np.isnan(data.t["feature_invalid"]))
    # #     self.assertTrue(np.isnan(data.U["feature_invalid"][0]))
    # #     self.assertTrue(np.isnan(data.U["feature_invalid"][1]))
    # #     self.assertTrue(np.isnan(data.U["feature_invalid"][2]))


    def test_is_regular_true(self):
        results = [{"TestingModel1d": {"values": np.arange(0, 10)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 10)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": None}},
                   {"TestingModel1d": {"values": np.arange(0, 10)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 10)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": None}},
                   {"TestingModel1d": {"values": np.arange(0, 10)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 10)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": None}}]

        self.assertTrue(self.runmodel.is_regular(results, "feature2d"))
        self.assertTrue(self.runmodel.is_regular(results, "feature1d"))
        self.assertTrue(self.runmodel.is_regular(results, "TestingModel1d"))


    def test_is_regular_nested_list(self):
        results = [{"test": {"values": [[0, 1], [1, 2], [3, 4]]}},
                   {"test": {"values": [[0, 1], [1, 2], [3, 4]]}},
                   {"test": {"values": [[0, 1], [1, 2], [3, 4]]}}]

        self.assertTrue(self.runmodel.is_regular(results, "test"))


    def test_is_regular_nested_nan(self):
        results = [{"test": {"values": [[0, 1], np.nan, [3, 4]]}},
                   {"test": {"values": [[0, 1], [1, 2], [3, 4]]}},
                   {"test": {"values": [[0, 1], [1, 2], [3, 4]]}}]

        self.assertTrue(self.runmodel.is_regular(results, "test"))


    def test_is_regular_false(self):
        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.runmodel = RunModel(model=TestingModelAdaptive(),
                                 parameters=self.parameters,
                                 features=features,
                                 logger_level="error")



        results = [{"TestingModelAdaptive": {"values": np.arange(0, 10)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                      np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 10)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": np.nan}},
                   {"TestingModelAdaptive": {"values": np.arange(0, 15)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 10)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": None}},
                   {"TestingModelAdaptive": {"values": np.arange(0, 10)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                      np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 15)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": None}}]


        self.assertTrue(self.runmodel.is_regular(results, "feature2d"))
        self.assertFalse(self.runmodel.is_regular(results, "TestingModelAdaptive"))
        self.assertFalse(self.runmodel.is_regular(results, "feature1d"))


    def test_is_regular_irregular_evaluation(self):
        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.runmodel = RunModel(model=TestingModelAdaptive(),
                                 parameters=self.parameters,
                                 features=features,
                                 logger_level="error")



        results = [{"TestingModelAdaptive": {"values": [1, [1, 2]]},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                      np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 10)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": None}},
                   {"TestingModelAdaptive": {"values": np.arange(0, 15)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                      np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 10)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": None}},
                   {"TestingModelAdaptive": {"values": np.arange(0, 10)},
                    "feature2d": {"values": np.array([np.arange(0, 10),
                                                      np.arange(0, 10)])},
                    "feature1d": {"values": np.arange(0, 15)},
                    "feature0d": {"values": 1},
                    "feature_invalid": {"values": np.nan}}]


        self.assertTrue(self.runmodel.is_regular(results, "feature2d"))
        self.assertFalse(self.runmodel.is_regular(results, "TestingModelAdaptive"))
        self.assertFalse(self.runmodel.is_regular(results, "feature1d"))


    def test_is_regular_nan(self):
        results = [{"test": {"values": np.nan}},
                   {"test": {"values": np.arange(0, 15)}},
                   {"test": {"values": np.arange(0, 10)}}]

        self.assertFalse(self.runmodel.is_regular(results, "test"))


    def test_is_regular_list(self):
        results = [{"test": {"values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}},
                   {"test": {"values": np.arange(0, 10)}},
                   {"test": {"values": np.arange(0, 10)}}]

        self.assertTrue(self.runmodel.is_regular(results, "test"))


    def test_is_regular_empty_list(self):
        results = [{"test": {"values": [0, [], 2, 3, 4, 5, 6, 7, 8, 9]}},
                   {"test": {"values": np.arange(0, 10)}},
                   {"test": {"values": np.arange(0, 10)}}]

        self.assertFalse(self.runmodel.is_regular(results, "test"))


    def test_apply_interpolation(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.model.interpolate = True

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertTrue(np.array_equal(results[0]["TestingModel1d"]["time"], results[1]["TestingModel1d"]["time"]))
        self.assertTrue(np.array_equal(results[1]["TestingModel1d"]["time"], results[2]["TestingModel1d"]["time"]))

        self.assertIsInstance(results[0]["TestingModel1d"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)
        self.assertIsInstance(results[1]["TestingModel1d"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)
        self.assertIsInstance(results[2]["TestingModel1d"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        time, interpolated_solves = self.runmodel.apply_interpolation(results, "TestingModel1d")

        self.assertTrue(np.array_equal(time, np.arange(0, 10)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 10) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 10) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 10) + 5.))

        results[1]["TestingModel1d"]["time"] = np.arange(0, 20)

        time, interpolated_solves = self.runmodel.apply_interpolation(results, "TestingModel1d")

        self.assertTrue(np.array_equal(time, np.arange(0, 20)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 20) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 20) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 20) + 5.))


    def test_apply_interpolation_none(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.model.interpolate = True

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertTrue(np.array_equal(results[0]["TestingModel1d"]["time"], results[1]["TestingModel1d"]["time"]))
        self.assertTrue(np.array_equal(results[1]["TestingModel1d"]["time"], results[2]["TestingModel1d"]["time"]))

        results[1]["TestingModel1d"]["interpolation"] = None

        time, interpolated_solves = self.runmodel.apply_interpolation(results, "TestingModel1d")

        self.assertTrue(np.array_equal(time, np.arange(0, 10)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 10) + 1))
        self.assertTrue(np.isnan(interpolated_solves[1]))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 10) + 5.))


    def test_run_two_uncertain_parameters(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=features,
                                 CPUs=1,
                                 logger_level="error")

        uncertain_parameters = ["a", "b"]

        data = self.runmodel.run(nodes, uncertain_parameters)

        features = list(data.keys())
        features.sort()
        self.assertEqual(features,
                         ["TestingModel1d", "feature0d", "feature1d", "feature2d"])

        self.assert_testingmodel1d(data)
        self.assert_feature_0d(data)
        self.assert_feature_1d(data)
        self.assert_feature_2d(data)


    def test_run_one_uncertain_parameter(self):
        nodes = np.array([0, 1, 2])
        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=None,
                                 CPUs=1,
                                 logger_level="error")

        uncertain_parameters = ["a"]

        data = self.runmodel.run(nodes, uncertain_parameters)

        self.assertIn("TestingModel1d", data)
        self.assertTrue(np.array_equal(data["TestingModel1d"]["time"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["TestingModel1d"].evaluations[0],
                                       np.arange(0, 10) + 2))
        self.assertTrue(np.array_equal(data["TestingModel1d"].evaluations[1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(data["TestingModel1d"].evaluations[2],
                                       np.arange(0, 10) + 4))


    def test_regularize_nan_results(self):
        results = [{"a": {"values": np.full(3, np.nan),
                          "time": np.full(3, np.nan)}},
                   {"a": {"values": np.full((3, 3, 3), 2),
                          "time": np.full((3, 3, 3), 2)}},
                   {"a": {"values": np.full(3, np.nan),
                          "time": np.full(3, np.nan)}}]


        new_results = self.runmodel.regularize_nan_results(results)

        correct_results = [{"a": {"values": np.full((3, 3, 3), np.nan),
                                  "time": np.full((3, 3, 3), np.nan)}},
                           {"a": {"values": np.full((3, 3, 3), 2),
                                  "time": np.full((3, 3, 3), 2)}},
                           {"a": {"values": np.full((3, 3, 3), np.nan),
                                  "time": np.full((3, 3, 3), np.nan)}}]

        self.assertEqual(correct_results[0]["a"]["values"].shape, new_results[0]["a"]["values"].shape)
        self.assertEqual(correct_results[1]["a"]["values"].shape, new_results[1]["a"]["values"].shape)
        self.assertEqual(correct_results[2]["a"]["values"].shape, new_results[2]["a"]["values"].shape)

        self.assertEqual(correct_results[0]["a"]["time"].shape, new_results[0]["a"]["time"].shape)
        self.assertEqual(correct_results[1]["a"]["time"].shape, new_results[1]["a"]["time"].shape)
        self.assertEqual(correct_results[2]["a"]["time"].shape, new_results[2]["a"]["time"].shape)





        results = [{"a": {"values": np.full(3, np.nan),
                          "time": np.nan}},
                   {"a": {"values": np.full((3, 3, 3), 2),
                          "time": np.nan}},
                   {"a": {"values": np.full(3, np.nan),
                          "time": np.nan}}]


        new_results = self.runmodel.regularize_nan_results(results)

        correct_results = [{"a": {"values": np.full((3, 3, 3), np.nan),
                                  "time": np.nan}},
                           {"a": {"values": np.full((3, 3, 3), 2),
                                  "time": np.nan}},
                           {"a": {"values": np.full((3, 3, 3), np.nan),
                                  "time": np.nan}}]

        self.assertEqual(correct_results[0]["a"]["values"].shape, new_results[0]["a"]["values"].shape)
        self.assertEqual(correct_results[1]["a"]["values"].shape, new_results[1]["a"]["values"].shape)
        self.assertEqual(correct_results[2]["a"]["values"].shape, new_results[2]["a"]["values"].shape)

        self.assertEqual(np.shape(correct_results[0]["a"]["time"]), np.shape(new_results[0]["a"]["time"]))
        self.assertEqual(np.shape(correct_results[1]["a"]["time"]), np.shape(new_results[1]["a"]["time"]))
        self.assertEqual(np.shape(correct_results[2]["a"]["time"]), np.shape(new_results[2]["a"]["time"]))

    # def test_regularize_nan_results_empty_list(self):
    #     results = [{"a": {"values": [1, 2, 4],
    #                       "time": 1}},
    #                {"a": {"values": [[], [1]],
    #                       "time": 2}},
    #                {"a": {"values": [1],
    #                       "time": 3}}]
    #     i = 0

    #     print results
    #     print "==========="

    #     new_results = self.runmodel.regularize_nan_results(results)
    #     print new_results

    #     correct_results = [{"a": {"values": np.full((3, 3, 3), np.nan),
    #                               "time": np.full((3, 3, 3), np.nan)}},
    #                        {"a": {"values": np.full((3, 3, 3), 2),
    #                               "time": np.full((3, 3, 3), 2)}},
    #                        {"a": {"values": np.full((3, 3, 3), np.nan),
    #                               "time": np.full((3, 3, 3), np.nan)}}]

    #     # self.assertEqual(correct_results[0]["a"]["values"].shape, new_results[0]["a"]["values"].shape)
    #     # self.assertEqual(correct_results[1]["a"]["values"].shape, new_results[1]["a"]["values"].shape)
    #     # self.assertEqual(correct_results[2]["a"]["values"].shape, new_results[2]["a"]["values"].shape)

    #     # self.assertEqual(correct_results[0]["a"]["time"].shape, new_results[0]["a"]["time"].shape)
    #     # self.assertEqual(correct_results[1]["a"]["time"].shape, new_results[1]["a"]["time"].shape)
    #     # self.assertEqual(correct_results[2]["a"]["time"].shape, new_results[2]["a"]["time"].shape)
