import unittest
import os
import shutil
import scipy.interpolate

import numpy as np

from xvfbwrapper import Xvfb
from uncertainpy.core.parallel import Parallel
from uncertainpy.models import NeuronModel, Model
from uncertainpy.features import Features

from .testing_classes import TestingFeatures
from .testing_classes import TestingModel1d, model_function
from .testing_classes import TestingModelNoTime
from .testing_classes import TestingModelAdaptive
from .testing_classes import PostprocessErrorNumpy
from .testing_classes import PostprocessErrorOne
from .testing_classes import PostprocessErrorValue



class TestParallel(unittest.TestCase):
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

        self.parallel = Parallel(model=TestingModel1d(),
                                 features=self.features)

        self.model_parameters = {"a": 0, "b": 1}

        self.t = np.arange(0, 10)
        self.values = np.arange(0, 10) + 1


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)




    def test_init(self):
        Parallel(TestingModel1d())


    def test_feature(self):
        self.parallel.features = Features
        self.assertIsInstance(self.parallel.features, Features)



    def test_feature_function(self):
        def feature_function(time, values):
                return "time", "values"

        self.parallel.features = feature_function
        self.assertIsInstance(self.parallel.features, Features)

        time, values = self.parallel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        self.assertEqual(self.parallel.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(time, values):
                return "time", "values"

        def feature_function2(time, values):
                return "time2", "values2"


        self.parallel.features = [feature_function, feature_function2]
        self.assertIsInstance(self.parallel.features, Features)

        time, values = self.parallel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")


        time, values = self.parallel.features.feature_function(None, None)
        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        time, values = self.parallel.features.feature_function2(None, None)
        self.assertEqual(time, "time2")
        self.assertEqual(values, "values2")

        self.assertEqual(self.parallel.features.features_to_run,
                         ["feature_function", "feature_function2"])



    def test_model(self):
        self.parallel.model = model_function
        self.assertIsInstance(self.parallel.model, Model)
        self.assertEqual(self.parallel.model.name, "model_function")




    # def test_sort_features(self):
    #     results = {"TestingModel1d": {"values": np.arange(0, 10) + 1,
    #                                     "time": np.arange(0, 10)},
    #                "feature1d": {"values": np.arange(0, 10),
    #                              "time": np.arange(0, 10)},
    #                "feature0d": {"values": 1,
    #                              "time": np.nan},
    #                "feature2d": {"values": np.array([np.arange(0, 10),
    #                                             np.arange(0, 10)]),
    #                              "time": np.arange(0, 10)},
    #                "feature_adaptive": {"values": np.arange(0, 10) + 1,
    #                                     "time": np.arange(0, 10),
    #                                     "interpolation": "interpolation object"},
    #                "feature_invalid": {"values": np.nan,
    #                                    "time": np.nan}}

    #     features_0d, features_1d, features_2d = self.parallel.sort_features(results)

    #     self.assertEqual(features_0d, ["feature0d", "feature_invalid"])
    #     self.assertEqual(set(features_1d),
    #                      set(["feature1d", "TestingModel1d", "feature_adaptive"]))
    #     self.assertEqual(features_2d, ["feature2d"])




    def test_create_interpolations(self):
        results = {"TestingModel1d": {"values": np.arange(0, 10) + 1,
                                      "time": np.arange(0, 10)},
                   "feature1d": {"values": np.arange(0, 10),
                                 "time": np.arange(0, 10)},
                   "feature0d": {"values": 1,
                                 "time": np.nan},
                   "feature2d": {"values": np.array([np.arange(0, 10),
                                                np.arange(0, 10)]),
                                 "time": np.arange(0, 10)},
                   "feature_adaptive": {"values": np.arange(0, 10) + 1,
                                        "time": np.arange(0, 10)},
                   "feature_invalid": {"values": np.nan,
                                       "time": np.nan}}

        results = self.parallel.create_interpolations(results)


        self.assertTrue(np.array_equal(results["TestingModel1d"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["TestingModel1d"]["values"], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(results["feature1d"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature1d"]["values"], np.arange(0, 10)))
        self.assertTrue(np.isnan(results["feature0d"]["time"]))
        self.assertEqual(results["feature0d"]["values"], 1)
        self.assertTrue(np.array_equal(results["feature2d"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature2d"]["values"], np.array([np.arange(0, 10),
                                                                            np.arange(0, 10)])))
        self.assertTrue(np.isnan(results["feature_invalid"]["time"]))
        self.assertTrue(np.isnan(results["feature_invalid"]["values"]))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["values"], np.arange(0, 10) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)



    def test_create_interpolations_feature_1d_no_t(self):
        results = {"feature_adaptive": {"values": np.arange(0, 10),
                                        "time": np.nan}}

        with self.assertRaises(AttributeError):
            self.parallel.create_interpolations(results)


    def test_create_interpolations_feature_0d(self):
        results = {"feature_adaptive": {"values": 1,
                                        "time": np.arange(0, 10)}}

        with self.assertRaises(AttributeError):
            self.parallel.create_interpolations(results)


    def test_create_interpolations_feature_2d(self):
        results = {"feature_adaptive": {"values": np.array([np.arange(0, 10),
                                                       np.arange(0, 10)]),
                                        "time": np.arange(0, 10)}}

        with self.assertRaises(NotImplementedError):
            self.parallel.create_interpolations(results)



    def test_create_interpolations_model_0d(self):
        self.parallel.model.adaptive = True
        results = {"TestingModel1d": {"values": 1,
                                      "time": np.arange(0, 10)}}

        with self.assertRaises(AttributeError):
            self.parallel.create_interpolations(results)


    def test_create_interpolations_model_2d(self):
        self.parallel.model.adaptive = True
        results = {"TestingModel1d": {"values": np.array([np.arange(0, 10),
                                                     np.arange(0, 10)]),
                                      "time": np.arange(0, 10)}}

        with self.assertRaises(NotImplementedError):
            self.parallel.create_interpolations(results)


    def test_run(self):
        results = self.parallel.run(self.model_parameters)

        self.assertTrue(self.parallel.features.is_preprocess_run)

        self.assertTrue(np.array_equal(results["TestingModel1d"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["TestingModel1d"]["values"], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(results["feature1d"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature1d"]["values"], np.arange(0, 10)))
        self.assertTrue(np.isnan(results["feature0d"]["time"]))
        self.assertEqual(results["feature0d"]["values"], 1)
        self.assertTrue(np.array_equal(results["feature2d"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature2d"]["values"], np.array([np.arange(0, 10),
                                                                            np.arange(0, 10)])))
        self.assertTrue(np.isnan(results["feature_invalid"]["time"]))
        self.assertTrue(np.isnan(results["feature_invalid"]["values"]))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["time"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["values"], np.arange(0, 10) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)


    def test_run_adaptive(self):
        parallel = Parallel(model=TestingModelAdaptive(),
                            features=TestingFeatures(features_to_run="feature_adaptive"))
        results = parallel.run(self.model_parameters)

        self.assertTrue(np.array_equal(results["TestingModelAdaptive"]["time"], np.arange(0, 11)))
        self.assertTrue(np.array_equal(results["TestingModelAdaptive"]["values"], np.arange(0, 11) + 1))
        self.assertIsInstance(results["TestingModelAdaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        self.assertTrue(np.array_equal(results["feature_adaptive"]["time"], np.arange(0, 11)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["values"], np.arange(0, 11) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)


    def test_run_model_no_time(self):
        parallel = Parallel(model=TestingModelNoTime())
        with self.assertRaises(ValueError):
            parallel.run(self.model_parameters)


    def test_run_feature_no_time(self):
        parallel = Parallel(model=TestingModel1d(),
                            features=TestingFeatures(features_to_run="feature_no_time"))

        with self.assertRaises(ValueError):
            parallel.run(self.model_parameters)


    def test_run_neuron_model(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "models/dLGN_modelDB/")

        model = NeuronModel(path=path,
                            adaptive=True)

        parallel = Parallel(model=model)
        model_parameters = {"cap": 1.1, "Rm": 22000}


        with Xvfb() as xvfb:
            parallel.run(model_parameters)


    def test_postprocess_error_numpy(self):
        parallel = Parallel(model=PostprocessErrorNumpy())

        with self.assertRaises(ValueError):
            parallel.run(self.model_parameters)


    def test_postprocess_error_one(self):
        parallel = Parallel(model=PostprocessErrorOne())

        with self.assertRaises(TypeError):
            parallel.run(self.model_parameters)


    def test_postprocess_error_value(self):
        parallel = Parallel(model=PostprocessErrorValue())

        with self.assertRaises(ValueError):
            parallel.run(self.model_parameters)


    def test_use_info_arg(self):
        def model_function(**model_parameters):
            return 1, 2, True

        def feature_function(time, values, info=False):
            self.assertTrue(info)

            return "time", "values"

        self.parallel.model = model_function
        self.parallel.features = feature_function

        self.parallel.run(self.model_parameters)


    def test_use_info_arg_dict(self):
        def model_function(**model_parameters):
            return 1, 2, {"1": 1, "2": 2}

        def feature_function(time, values, info):
            self.assertEqual(info["1"], 1)
            self.assertEqual(info["2"], 2)

            return "time", "values"

        self.parallel.model = model_function
        self.parallel.features = feature_function

        self.parallel.run(self.model_parameters)


    def test_use_model_feature_arguments_error(self):
        def model_function(**model_parameters):
            return 1, 2, 3

        def feature_function(time, values):
            return "time", "values"

        self.parallel.model = model_function
        self.parallel.features = feature_function
        with self.assertRaises(TypeError):
            self.parallel.run(self.model_parameters)



    def test_none_to_nan(self):

        values_irregular = np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])])

        result = self.parallel.none_to_nan(values_irregular)

        values_correct = np.array([[np.nan, np.nan, np.nan], [1, 2, 3],
                              [np.nan, np.nan, np.nan], [1, 2, 3]])


        result = np.array(result)
        self.assertTrue(((result == values_correct) | (np.isnan(result) & np.isnan(values_correct))).all())



        values_irregular = np.array([None,
                                np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
                                np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
                                np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
                                None])

        result = self.parallel.none_to_nan(values_irregular)

        values_correct = np.array([[[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
                               [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                              [[np.nan, np.nan, np.nan], [1, 2, 3],
                               [np.nan, np.nan, np.nan], [1, 2, 3]],
                              [[np.nan, np.nan, np.nan], [1, 2, 3],
                               [np.nan, np.nan, np.nan], [1, 2, 3]],
                              [[np.nan, np.nan, np.nan], [1, 2, 3],
                               [np.nan, np.nan, np.nan], [1, 2, 3]],
                              [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
                               [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]])

        result = np.array(result)
        self.assertTrue(((result == values_correct) | (np.isnan(result) & np.isnan(values_correct))).all())



        values_irregular = np.array([np.array([1, 2, 3]), np.array([1, 2, 3]),
                                np.array([1, 2, 3]), np.array([1, 2, 3])])

        result = self.parallel.none_to_nan(values_irregular)

        result = np.array(result)
        self.assertTrue(np.array_equal(result, values_irregular))



        values_irregular = np.array([None, np.array([np.array(1), np.array(2), np.array(3)]),
                                None, np.array([np.array(1), np.array(2), np.array(3)])])

        result = self.parallel.none_to_nan(values_irregular)

        values_correct = np.array([[np.nan, np.nan, np.nan], [1, 2, 3],
                              [np.nan, np.nan, np.nan], [1, 2, 3]])

        result = np.array(result)
        self.assertTrue(((result == values_correct) | (np.isnan(result) & np.isnan(values_correct))).all())



        values_irregular = np.array([np.array(1), np.array(2), np.array(3)])

        result = self.parallel.none_to_nan(values_irregular)

        values_correct = np.array([np.array(1), np.array(2), np.array(3)])

        result = np.array(result)
        self.assertTrue(np.array_equal(result, values_irregular))


        values_irregular = np.array([None, None, None])

        result = self.parallel.none_to_nan(values_irregular)

        values_correct = np.array([np.nan, np.nan, np.nan])

        result = np.array(result)

        self.assertTrue(np.all(np.isnan(result)))
        self.assertEqual(len(result), 3)
