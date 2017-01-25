import unittest
import shutil
import os
import subprocess

import numpy as np

from uncertainpy import Data

class TestData(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


        self.data = Data()


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)



    def test_sortFeatures(self):
        test_result = {"directComparison": (None, np.arange(0, 10), None),
                       "feature2d": (None, np.array([np.arange(0, 10),
                                                     np.arange(0, 10)]), None),
                       "feature1d": (None, np.arange(0, 10), None),
                       "feature0d": (None, 1, None),
                       "featureInvalid": (None, np.nan, None)}

        features_0d, features_1d, features_2d = self.data.sortFeatures(test_result)

        self.assertIn("directComparison", features_1d)
        self.assertIn("feature2d", features_2d)
        self.assertIn("feature1d", features_1d)
        self.assertIn("feature0d", features_0d)
        self.assertIn("featureInvalid", features_0d)




    def test_setFeatures(self):
        test_result = {"directComparison": (None, np.arange(0, 10), None),
                       "feature2d": (None, np.array([np.arange(0, 10),
                                                     np.arange(0, 10)]), None),
                       "feature1d": (None, np.arange(0, 10), None),
                       "feature0d": (None, 1, None),
                       "featureInvalid": (None, np.nan, None)}

        self.data.setFeatures(test_result)

        self.assertIn("directComparison", self.data.features_1d)
        self.assertIn("feature2d", self.data.features_2d)
        self.assertIn("feature1d", self.data.features_1d)
        self.assertIn("feature0d", self.data.features_0d)
        self.assertIn("featureInvalid", self.data.features_0d)


        self.assertEqual(set(self.data.feature_list), set(["directComparison",
                                                           "feature0d",
                                                           "feature1d",
                                                           "feature2d",
                                                           "featureInvalid"]))


    def test_isAdaptiveFalse(self):
        self.data.U = {"feature1": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)],
                       "directComparison": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)]}

        self.data.features_1d = ["feature1", "directComparison"]

        self.assertFalse(self.data.isAdaptive())


    def test_isAdaptiveTrue(self):
        self.data.U = {"feature1": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 5)],
                       "directComparison": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)]}

        self.data.features_1d = ["feature1", "directComparison"]

        self.assertTrue(self.data.isAdaptive())

    def test_save(self):
        self.data.t = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.U = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.E = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.Var = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.p_05 = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.p_95 = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.sensitivity_1 = {"feature1": [1, 2], "directComparison": [3., 4.]}
        self.data.total_sensitivity_1 = {"feature1": [1, 2], "directComparison": [3., 4.]}

        self.data.sensitivity_t = {"feature1": [1, 2], "directComparison": [3., 4.]}
        self.data.total_sensitivity_t = {"feature1": [1, 2], "directComparison": [3., 4.]}


        self.data.uncertain_parameters = ["a", "b"]
        self.data.xlabel = "xlabel"
        self.data.ylabel = "ylabel"
        self.data.feature_list = ["directComparison", "feature1"]


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock")
        filename = os.path.join(self.output_test_dir, "test_save_mock")

        self.data.save(filename)

        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)




    def test_load(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock")


        self.data.load(compare_file)

        self.assertTrue(np.array_equal(self.data.U["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.U["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.E["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.E["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.t["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.t["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.Var["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.Var["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.p_05["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.p_05["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.p_95["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.p_95["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.sensitivity_1["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.sensitivity_1["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.total_sensitivity_1["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.total_sensitivity_1["directComparison"], [3., 4.]))

        self.assertTrue(np.array_equal(self.data.sensitivity_t["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.sensitivity_t["directComparison"], [3., 4.]))
        self.assertTrue(np.array_equal(self.data.total_sensitivity_t["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.total_sensitivity_t["directComparison"], [3., 4.]))


        self.assertEqual(self.data.uncertain_parameters[0], "a")
        self.assertEqual(self.data.uncertain_parameters[1], "b")

        self.assertEqual(self.data.xlabel, "xlabel")
        self.assertEqual(self.data.ylabel, "ylabel")

        self.assertEqual(self.data.feature_list[0], "directComparison")
        self.assertEqual(self.data.feature_list[1], "feature1")


    def test_removeOnlyInvalidResultsNo(self):
        self.data.t = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.U = {"feature1": [1., 2.], "directComparison": [3., np.nan]}

        self.data.feature_list = ["directComparison", "feature1"]
        self.data.features_1d = ["directComparison", "feature1"]

        self.data.removeOnlyInvalidResults()


        self.assertTrue(np.array_equal(self.data.U["feature1"], [1., 2.]))
        self.assertEqual(self.data.U["directComparison"][0], 3.)
        self.assertTrue(np.isnan(self.data.U["directComparison"][1]))
        self.assertTrue(np.array_equal(self.data.t["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.t["directComparison"], [3., 4.]))

        self.assertEqual(self.data.feature_list[0], "directComparison")
        self.assertEqual(self.data.feature_list[1], "feature1")

        self.assertEqual(self.data.features_1d[0], "directComparison")
        self.assertEqual(self.data.features_1d[1], "feature1")


    def test_removeOnlyInvalidResultsError(self):
        self.data.t = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.U = {"feature1": [1., 2.], "directComparison": np.array([np.nan, np.nan])}


        self.data.feature_list = ["directComparison", "feature1"]
        self.data.features_1d = ["directComparison", "feature1"]

        with self.assertRaises(RuntimeWarning):
            self.data.removeOnlyInvalidResults()


        # self.assertTrue(np.array_equal(self.data.U["feature1"], [1., 2.]))
        # self.assertEqual(self.data.U["directComparison"], "Only invalid results for all set of parameters")
        # self.assertTrue(np.array_equal(self.data.t["feature1"], [1., 2.]))
        # self.assertTrue(np.array_equal(self.data.t["directComparison"], [3., 4.]))
        #
        # self.assertEqual(self.data.feature_list, ["feature1"])
        #
        # self.assertEqual(self.data.features_1d, ["feature1"])


    def test_resetValues(self):
        self.uncertain_parameters = None

        self.data.features_0d = -1
        self.data.features_1d = -1
        self.data.features_2d = -1
        self.data.feature_list = -1

        self.data.U = -1
        self.data.t = -1
        self.data.E = -1
        self.data.Var = -1
        self.data.p_05 = -1
        self.data.p_95 = -1
        self.data.sensitivity_1 = -1
        self.data.total_sensitivity_1 = -1
        self.data.sensitivity_t = -1
        self.data.total_sensitivity_t = -1


        self.data.xlabel = -1
        self.data.ylabel = -1

        self.data.resetValues()


        self.assertEqual(self.data.features_0d, [])

        self.assertEqual(self.data.features_1d, [])
        self.assertEqual(self.data.features_2d, [])
        self.assertEqual(self.data.feature_list, [])
        self.assertEqual(self.data.U, {})
        self.assertEqual(self.data.t, {})
        self.assertEqual(self.data.E, {})
        self.assertEqual(self.data.Var, {})
        self.assertEqual(self.data.p_05, {})
        self.assertEqual(self.data.p_95, {})
        self.assertEqual(self.data.sensitivity_1, {})
        self.assertEqual(self.data.total_sensitivity_1, {})
        self.assertEqual(self.data.sensitivity_t, {})
        self.assertEqual(self.data.total_sensitivity_t, {})
