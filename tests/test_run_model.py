import unittest
import os
import shutil
import scipy.interpolate

import numpy as np

from uncertainpy import RunModel, Parameters
from uncertainpy.models import NeuronModel, Model
from uncertainpy.features import GeneralFeatures, SpikingFeatures

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
                                                         "feature_adaptive"])

        self.parameterlist = [["a", 1, None],
                              ["b", 2, None]]

        self.parameters = Parameters(self.parameterlist)

        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=self.features,
                                 suppress_model_graphics=True)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        RunModel(model=TestingModel1d(), parameters=self.parameters)


    def test_set_feature(self):
        self.runmodel.features = GeneralFeatures()
        self.assertIsInstance(self.runmodel._features, GeneralFeatures)
        self.assertIsInstance(self.runmodel.parallel.features, GeneralFeatures)



    def test_set_model(self):
        self.runmodel = RunModel(TestingModel2d(), None)
        self.runmodel.model = TestingModel1d()

        self.assertIsInstance(self.runmodel._model, TestingModel1d)
        self.assertIsInstance(self.runmodel.parallel.model, TestingModel1d)


    def test_set_model_none(self):
        self.runmodel = RunModel(TestingModel2d(), None)
        self.runmodel.model = None

        self.assertIsNone(self.runmodel._model)
        self.assertIsNone(self.runmodel.parallel.model)


    def test_set_parameter_list(self):
        runmodel = RunModel(TestingModel2d(), None)

        runmodel.parameters = self.parameterlist

        self.assertIsInstance(runmodel.parameters, Parameters)


    def test_set_parameter_error(self):
        runmodel = RunModel(TestingModel2d(), None)

        with self.assertRaises(TypeError):
            runmodel.parameters = 2


    def test_set_model_function(self):
        self.runmodel = RunModel(TestingModel2d(), None)

        self.runmodel.model = model_function

        self.assertIsInstance(self.runmodel.model, Model)
        self.assertIsInstance(self.runmodel.parallel.model, Model)


    def test_init_model_function(self):
        self.runmodel = RunModel(model_function, None)

        self.assertIsInstance(self.runmodel.model, Model)
        self.assertIsInstance(self.runmodel.parallel.model, Model)


    def test_feature_function(self):
        def feature_function(t, U):
            return "t", "U"

        self.runmodel.features = feature_function
        self.assertIsInstance(self.runmodel.features, GeneralFeatures)

        t, U = self.runmodel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        self.assertEqual(self.runmodel.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(t, U):
            return "t", "U"

        def feature_function2(t, U):
            return "t2", "U2"


        self.runmodel.features = [feature_function, feature_function2]
        self.assertIsInstance(self.runmodel.features, GeneralFeatures)

        t, U = self.runmodel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")


        t, U = self.runmodel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.runmodel.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")


        t, U = self.runmodel.parallel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.runmodel.parallel.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

        self.assertEqual(self.runmodel.features.features_to_run,
                         ["feature_function", "feature_function2"])


    def test_feature_functions_base(self):
        def feature_function(t, U):
            return "t", "U"

        def feature_function2(t, U):
            return "t2", "U2"

        implemented_features = ["nr_spikes", "time_before_first_spike",
                                "spike_rate", "average_AP_overshoot",
                                "average_AHP_depth", "average_AP_width",
                                "accomondation_index"]

        self.runmodel.features = SpikingFeatures([feature_function, feature_function2])
        self.assertIsInstance(self.runmodel.features, SpikingFeatures)

        t, U = self.runmodel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")


        t, U = self.runmodel.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

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
                                 CPUs=1)

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
                                 CPUs=3)


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
                              "TestingModel1d", "feature_invalid", "feature_adaptive"]))


    def test_evaluate_nodes_parallel_model_1d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.CPUs = 3

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel1d", "feature_invalid", "feature_adaptive"]))

    def test_evaluate_nodes_sequential_model_2d(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_invalid"])

        self.runmodel = RunModel(model=TestingModel2d(),
                                 parameters=self.parameters,
                                 features=features,
                                 CPUs=1)

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
                                 CPUs=3)

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        self.assertEqual(set(results[0].keys()),
                         set(["feature0d", "feature1d", "feature2d",
                              "TestingModel2d", "feature_invalid"]))

    def test_evaluate_nodes_not_supress_graphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.suppress_model_graphics = False
        self.runmodel.evaluate_nodes(nodes, ["a", "b"])


    def test_evaluate_nodes_supress_graphics(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.supress_model_suppress_model_graphics = True
        self.runmodel.evaluate_nodes(nodes, ["a", "b"])



    def test_results_to_data_model_1d_all_features(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=features,
                                 suppress_model_graphics=True)

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        data = self.runmodel.results_to_data(results)

        features = data.keys()
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
                                 suppress_model_graphics=True)

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        data = self.runmodel.results_to_data(results)

        self.assertNotIn("feature_invalid", data)


    def test_results_to_data_model_1d_features_all_adaptive(self):

        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d",
                                                    "feature_adaptive"],
                                   adaptive="feature_adaptive")

        self.runmodel = RunModel(model=TestingModelAdaptive(),
                                 parameters=self.parameters,
                                 features=features,
                                 suppress_model_graphics=True)


        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])

        data = self.runmodel.results_to_data(results)

        features = data.keys()
        features.sort()

        self.assertEqual(features,
                         ["TestingModelAdaptive", "feature0d", "feature1d",
                          "feature2d", "feature_adaptive"])

        self.assertIn("TestingModelAdaptive", data)

        self.assertTrue(np.array_equal(data["TestingModelAdaptive"]["t"],
                                       np.arange(0, 15)))
        self.assertTrue(np.allclose(data["TestingModelAdaptive"]["U"][0],
                                    np.arange(0, 15) + 1))
        self.assertTrue(np.allclose(data["TestingModelAdaptive"]["U"][1],
                                    np.arange(0, 15) + 3))
        self.assertTrue(np.allclose(data["TestingModelAdaptive"]["U"][2],
                                    np.arange(0, 15) + 5))


        self.assertTrue(np.array_equal(data["feature_adaptive"]["t"],
                                       np.arange(0, 15)))
        self.assertTrue(np.allclose(data["feature_adaptive"]["U"][0],
                                    np.arange(0, 15) + 1))
        self.assertTrue(np.allclose(data["feature_adaptive"]["U"][1],
                                    np.arange(0, 15) + 3))
        self.assertTrue(np.allclose(data["feature_adaptive"]["U"][2],
                                    np.arange(0, 15) + 5))


        self.assert_feature_0d(data)
        self.assert_feature_1d(data)
        self.assert_feature_2d(data)



    # # def test_results_to_dataAdaptiveError(self):
    # #     self.runmodel = RunModel(TestingModelAdaptive(adaptive=True),
    # #                              features=TestingFeatures(),
    # #                              supress_model_output=True,
    # #                              suppress_model_graphics=True)
    # #
    # #
    # #     nodes = np.array([[0, 1, 2], [1, 2, 3]])
    # #     self.runmodel.data.uncertain_parameters = ["a", "b"]
    # #
    # #
    # #     results = self.runmodel.evaluate_nodes(nodes)
    # #     self.runmodel.results_to_data(results)
    # #
    # #     self.assertEqual(set(self.runmodel.data.U.keys()),
    # #                      set(["feature0d", "feature1d", "feature2d",
    # #                           "TestingModelAdaptive", "feature_invalid"]))
    # #
    # #     self.assertEqual(set(self.runmodel.data.t.keys()),
    # #                      set(["feature0d", "feature1d", "feature2d",
    # #                           "TestingModelAdaptive", "feature_invalid"]))
    # #
    # #     self.assertEqual(set(self.runmodel.data.feature_list),
    # #                      set(["feature0d", "feature1d", "feature2d",
    # #                           "TestingModelAdaptive", "feature_invalid"]))
    # #
    # #     self.assertIn("TestingModelAdaptive", self.runmodel.data.U.keys())
    # #     self.assertTrue(np.array_equal(self.runmodel.data.t["TestingModelAdaptive"],
    # #                                    np.arange(0, 10)))
    # #     self.assertTrue(np.allclose(self.runmodel.data.U["TestingModelAdaptive"][0],
    # #                                 np.arange(0, 10) + 1))
    # #     self.assertTrue(np.allclose(self.runmodel.data.U["TestingModelAdaptive"][1],
    # #                                 np.arange(0, 10) + 3))
    # #     self.assertTrue(np.allclose(self.runmodel.data.U["TestingModelAdaptive"][2],
    # #                                 np.arange(0, 10) + 5))
    # #
    # #     self.assert_feature_0d(self.runmodel.data)
    # #     self.assert_feature_1d(self.runmodel.data)
    # #     self.assert_feature_2d(self.runmodel.data)
    # #     self.assert_feature_invalid(self.runmodel.data)




    def assert_feature_0d(self, data):
        self.assertIn("feature0d", data)
        self.assertNotIn("t", data["feature0d"])
        self.assertTrue(np.array_equal(data["feature0d"]["U"], [1, 1, 1]))
        self.assertEqual(data["feature0d"]["labels"], ["feature0d"])


    def assert_feature_1d(self, data):
        self.assertIn("feature1d", data)

        self.assertTrue(np.array_equal(data["feature1d"]["t"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature1d"]["U"][0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature1d"]["U"][1],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature1d"]["U"][2],
                                       np.arange(0, 10)))
        self.assertEqual(data["feature1d"]["labels"], ["feature1d x", "feature1d y"])



    def assert_feature_2d(self, data):
        self.assertIn("feature2d", data)
        self.assertTrue(np.array_equal(data["feature2d"]["t"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["feature2d"]["U"][0],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(data["feature2d"]["U"][1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertTrue(np.array_equal(data["feature2d"]["U"][2],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertEqual(data["feature2d"]["labels"], ["feature2d x",
                                                       "feature2d y",
                                                       "feature2d z"])


    def assert_testingmodel1d(self, data):
        self.assertIn("TestingModel1d", data)
        self.assertTrue(np.array_equal(data["TestingModel1d"]["t"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["TestingModel1d"]["U"][0],
                                       np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(data["TestingModel1d"]["U"][1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(data["TestingModel1d"]["U"][2],
                                       np.arange(0, 10) + 5))
        self.assertEqual(data["TestingModel1d"]["labels"], ["x", "y"])


    # def assert_feature_invalid(self, data):
    #     self.assertIn("feature_invalid", data.U.keys())
    #     self.assertTrue(np.isnan(data.t["feature_invalid"]))
    #     self.assertTrue(np.isnan(data.U["feature_invalid"][0]))
    #     self.assertTrue(np.isnan(data.U["feature_invalid"][1]))
    #     self.assertTrue(np.isnan(data.U["feature_invalid"][2]))


    def test_is_adaptive_false(self):
        results = [{"TestingModel1d": {"U": np.arange(0, 10)},
                    "feature2d": {"U": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"U": np.arange(0, 10)},
                    "feature0d": {"U": 1},
                    "feature_invalid": {"U": None}},
                   {"TestingModel1d": {"U": np.arange(0, 10)},
                    "feature2d": {"U": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"U": np.arange(0, 10)},
                    "feature0d": {"U": 1},
                    "feature_invalid": {"U": None}},
                   {"TestingModel1d": {"U": np.arange(0, 10)},
                    "feature2d": {"U": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"U": np.arange(0, 10)},
                    "feature0d": {"U": 1},
                    "feature_invalid": {"U": None}}]

        self.assertFalse(self.runmodel.is_adaptive(results, "feature2d"))
        self.assertFalse(self.runmodel.is_adaptive(results, "feature1d"))
        self.assertFalse(self.runmodel.is_adaptive(results, "TestingModel1d"))



    def test_is_adaptive_true(self):
        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.runmodel = RunModel(model=TestingModelAdaptive(),
                                 parameters=self.parameters,
                                 features=features,
                                 suppress_model_graphics=True)



        results = [{"TestingModelAdaptive": {"U": np.arange(0, 10)},
                    "feature2d": {"U": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"U": np.arange(0, 10)},
                    "feature0d": {"U": 1},
                    "feature_invalid": {"U": None}},
                   {"TestingModelAdaptive": {"U": np.arange(0, 15)},
                    "feature2d": {"U": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"U": np.arange(0, 10)},
                    "feature0d": {"U": 1},
                    "feature_invalid": {"U": None}},
                   {"TestingModelAdaptive": {"U": np.arange(0, 10)},
                    "feature2d": {"U": np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])},
                    "feature1d": {"U": np.arange(0, 15)},
                    "feature0d": {"U": 1},
                    "feature_invalid": {"U": None}}]


        self.assertFalse(self.runmodel.is_adaptive(results, "feature2d"))
        self.assertTrue(self.runmodel.is_adaptive(results, "TestingModelAdaptive"))
        self.assertTrue(self.runmodel.is_adaptive(results, "feature1d"))


    def test_apply_interpolation(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        self.runmodel.model.adaptive = True

        results = self.runmodel.evaluate_nodes(nodes, ["a", "b"])
        ts = []
        interpolation = []

        for solved in results:
            ts.append(solved["TestingModel1d"]["t"])
            interpolation.append(solved["TestingModel1d"]["interpolation"])

        self.assertTrue(np.array_equal(ts[0], ts[1]))
        self.assertTrue(np.array_equal(ts[1], ts[2]))

        self.assertIsInstance(interpolation[0],
                              scipy.interpolate.fitpack2.UnivariateSpline)
        self.assertIsInstance(interpolation[1],
                              scipy.interpolate.fitpack2.UnivariateSpline)
        self.assertIsInstance(interpolation[2],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        t, interpolated_solves = self.runmodel.apply_interpolation(ts, interpolation)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 10) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 10) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 10) + 5.))

        ts[1] = np.arange(0, 20)

        t, interpolated_solves = self.runmodel.apply_interpolation(ts, interpolation)

        self.assertTrue(np.array_equal(t, np.arange(0, 20)))
        self.assertTrue(np.allclose(interpolated_solves[0],
                                    np.arange(0, 20) + 1))
        self.assertTrue(np.allclose(interpolated_solves[1],
                                    np.arange(0, 20) + 3.))
        self.assertTrue(np.allclose(interpolated_solves[2],
                                    np.arange(0, 20) + 5.))



    def test_run_two_uncertain_parameters(self):
        nodes = np.array([[0, 1, 2], [1, 2, 3]])
        features = TestingFeatures(features_to_run=["feature0d",
                                                    "feature1d",
                                                    "feature2d"])

        self.runmodel = RunModel(model=TestingModel1d(),
                                 parameters=self.parameters,
                                 features=features,
                                 CPUs=1)
        uncertain_parameters = ["a", "b"]

        data = self.runmodel.run(nodes, uncertain_parameters)

        features = data.keys()
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
                                 CPUs=1)
        uncertain_parameters = ["a"]

        data = self.runmodel.run(nodes, uncertain_parameters)

        self.assertIn("TestingModel1d", data)
        self.assertTrue(np.array_equal(data["TestingModel1d"]["t"],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(data["TestingModel1d"]["U"][0],
                                       np.arange(0, 10) + 2))
        self.assertTrue(np.array_equal(data["TestingModel1d"]["U"][1],
                                       np.arange(0, 10) + 3))
        self.assertTrue(np.array_equal(data["TestingModel1d"]["U"][2],
                                       np.arange(0, 10) + 4))


    def test_run_neuron_model(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "models/dLGN_modelDB/")

        model = NeuronModel(path=path,
                            adaptive=True)

        self.runmodel = RunModel(model=model, parameters=self.parameters, CPUs=1)
        uncertain_parameters = ["cap", "Rm"]
        nodes = np.array([[1.0, 1.1, 1.2], [21900, 22000, 22100]])

        self.runmodel.run(nodes, uncertain_parameters)
