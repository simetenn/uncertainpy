import numpy as np
import unittest
import scipy.interpolate
import os
import shutil

from xvfbwrapper import Xvfb
from uncertainpy import Parallel
from uncertainpy.models import NeuronModel, Model
from uncertainpy.features import GeneralFeatures, SpikingFeatures

from testing_classes import TestingFeatures
from testing_classes import TestingModel1d, model_function
from testing_classes import TestingModelNoTime
from testing_classes import TestingModelAdaptive


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
        self.U = np.arange(0, 10) + 1


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)




    def test_init(self):
        Parallel(TestingModel1d())


    def test_feature(self):
        self.parallel.features = GeneralFeatures
        self.assertIsInstance(self.parallel.features, GeneralFeatures)



    def test_feature_function(self):
        def feature_function(t, U):
                return "t", "U"

        self.parallel.features = feature_function
        self.assertIsInstance(self.parallel.features, GeneralFeatures)

        t, U = self.parallel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        self.assertEqual(self.parallel.features.features_to_run,
                         ["feature_function"])


    def test_feature_functions(self):
        def feature_function(t, U):
                return "t", "U"

        def feature_function2(t, U):
                return "t2", "U2"


        self.parallel.features = [feature_function, feature_function2]
        self.assertIsInstance(self.parallel.features, GeneralFeatures)

        t, U = self.parallel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")


        t, U = self.parallel.features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = self.parallel.features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

        self.assertEqual(self.parallel.features.features_to_run,
                         ["feature_function", "feature_function2"])



    def test_model(self):
        self.parallel.model = model_function
        self.assertIsInstance(self.parallel.model, Model)
        self.assertEqual(self.parallel.model.name, "model_function")




    def test_sort_features(self):
        results = {"TestingModel1d": {"U": np.arange(0, 10) + 1,
                                        "t": np.arange(0, 10)},
                   "feature1d": {"U": np.arange(0, 10),
                                 "t": np.arange(0, 10)},
                   "feature0d": {"U": 1,
                                 "t": np.nan},
                   "feature2d": {"U": np.array([np.arange(0, 10),
                                                np.arange(0, 10)]),
                                 "t": np.arange(0, 10)},
                   "feature_adaptive": {"U": np.arange(0, 10) + 1,
                                        "t": np.arange(0, 10),
                                        "interpolation": "interpolation object"},
                   "feature_invalid": {"U": np.nan,
                                       "t": np.nan}}

        features_0d, features_1d, features_2d = self.parallel.sort_features(results)

        self.assertEqual(features_0d, ["feature0d", "feature_invalid"])
        self.assertEqual(set(features_1d),
                         set(["feature1d", "TestingModel1d", "feature_adaptive"]))
        self.assertEqual(features_2d, ["feature2d"])




    def test_create_interpolations(self):
        results = {"TestingModel1d": {"U": np.arange(0, 10) + 1,
                                      "t": np.arange(0, 10)},
                   "feature1d": {"U": np.arange(0, 10),
                                 "t": np.arange(0, 10)},
                   "feature0d": {"U": 1,
                                 "t": np.nan},
                   "feature2d": {"U": np.array([np.arange(0, 10),
                                                np.arange(0, 10)]),
                                 "t": np.arange(0, 10)},
                   "feature_adaptive": {"U": np.arange(0, 10) + 1,
                                        "t": np.arange(0, 10)},
                   "feature_invalid": {"U": np.nan,
                                       "t": np.nan}}

        results = self.parallel.create_interpolations(results)


        self.assertTrue(np.array_equal(results["TestingModel1d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["TestingModel1d"]["U"], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(results["feature1d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature1d"]["U"], np.arange(0, 10)))
        self.assertTrue(np.isnan(results["feature0d"]["t"]))
        self.assertEqual(results["feature0d"]["U"], 1)
        self.assertTrue(np.array_equal(results["feature2d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature2d"]["U"], np.array([np.arange(0, 10),
                                                                            np.arange(0, 10)])))
        self.assertTrue(np.isnan(results["feature_invalid"]["t"]))
        self.assertTrue(np.isnan(results["feature_invalid"]["U"]))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["U"], np.arange(0, 10) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)



    def test_create_interpolations_feature_1d_no_t(self):
        results = {"feature_adaptive": {"U": np.arange(0, 10),
                                        "t": np.nan}}

        with self.assertRaises(AttributeError):
            self.parallel.create_interpolations(results)


    def test_create_interpolations_feature_0d(self):
        results = {"feature_adaptive": {"U": 1,
                                        "t": np.arange(0, 10)}}

        with self.assertRaises(AttributeError):
            self.parallel.create_interpolations(results)


    def test_create_interpolations_feature_2d(self):
        results = {"feature_adaptive": {"U": np.array([np.arange(0, 10),
                                                       np.arange(0, 10)]),
                                        "t": np.arange(0, 10)}}

        with self.assertRaises(NotImplementedError):
            self.parallel.create_interpolations(results)



    def test_create_interpolations_model_0d(self):
        self.parallel.model.adaptive = True
        results = {"TestingModel1d": {"U": 1,
                                      "t": np.arange(0, 10)}}

        with self.assertRaises(AttributeError):
            self.parallel.create_interpolations(results)


    def test_create_interpolations_model_2d(self):
        self.parallel.model.adaptive = True
        results = {"TestingModel1d": {"U": np.array([np.arange(0, 10),
                                                     np.arange(0, 10)]),
                                      "t": np.arange(0, 10)}}

        with self.assertRaises(NotImplementedError):
            self.parallel.create_interpolations(results)


    def test_run(self):
        results = self.parallel.run(self.model_parameters)

        self.assertTrue(self.parallel.features.is_preprocess_run)

        self.assertTrue(np.array_equal(results["TestingModel1d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["TestingModel1d"]["U"], np.arange(0, 10) + 1))
        self.assertTrue(np.array_equal(results["feature1d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature1d"]["U"], np.arange(0, 10)))
        self.assertTrue(np.isnan(results["feature0d"]["t"]))
        self.assertEqual(results["feature0d"]["U"], 1)
        self.assertTrue(np.array_equal(results["feature2d"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature2d"]["U"], np.array([np.arange(0, 10),
                                                                            np.arange(0, 10)])))
        self.assertTrue(np.isnan(results["feature_invalid"]["t"]))
        self.assertTrue(np.isnan(results["feature_invalid"]["U"]))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["t"], np.arange(0, 10)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["U"], np.arange(0, 10) + 1))
        self.assertIsInstance(results["feature_adaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)


    def test_run_adaptive(self):
        parallel = Parallel(model=TestingModelAdaptive(),
                            features=TestingFeatures(features_to_run="feature_adaptive"))
        results = parallel.run(self.model_parameters)

        self.assertTrue(np.array_equal(results["TestingModelAdaptive"]["t"], np.arange(0, 11)))
        self.assertTrue(np.array_equal(results["TestingModelAdaptive"]["U"], np.arange(0, 11) + 1))
        self.assertIsInstance(results["TestingModelAdaptive"]["interpolation"],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        self.assertTrue(np.array_equal(results["feature_adaptive"]["t"], np.arange(0, 11)))
        self.assertTrue(np.array_equal(results["feature_adaptive"]["U"], np.arange(0, 11) + 1))
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
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "models/dLGN_modelDB/")

        model = NeuronModel(model_path=model_path,
                            adaptive=True)

        parallel = Parallel(model=model)
        model_parameters = {"cap": 1.1, "Rm": 22000}


        with Xvfb() as xvfb:
            parallel.run(model_parameters)


    def test_to_array(self):
        U_irregular = np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])])

        U_irregular = np.array([np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
                                np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
                                np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
                                None])

        result = self.parallel.to_array(U_irregular)
        print result
