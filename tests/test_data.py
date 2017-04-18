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


    def test_features_0d(self):
        self.data.features_0d = ["feature0d"]

        self.assertEqual(self.data.feature_list, ["feature0d"])
        self.assertEqual(self.data.features_0d, ["feature0d"])

    def test_features_1d(self):
        self.data.features_1d = ["feature1d"]

        self.assertEqual(self.data.feature_list, ["feature1d"])
        self.assertEqual(self.data.features_1d, ["feature1d"])


    def test_nan_to_none(self):
        a = np.array([0, 1, 2, None, 4, None, None])
        b = np.array([0, 1, 2, np.nan, 4, np.nan, np.nan])

        result = self.data.nan_to_none(b)

        self.assertTrue(np.array_equal(a, result))


    def test_none_to_nan(self):
        a = [0, 1, 2, None, 4, None, None]
        b = np.array([0, 1, 2, np.nan, 4, np.nan, np.nan])

        result = self.data.none_to_nan(a)

        self.assertTrue(np.array_equal(b[~np.isnan(b)], result[~np.isnan(result)]))
        self.assertTrue(np.array_equal(np.isnan(b), np.isnan(result)))

    def test_features_2d(self):
        self.data.features_2d = ["feature2d"]

        self.assertEqual(self.data.feature_list, ["feature2d"])
        self.assertEqual(self.data.features_2d, ["feature2d"])


    def test_update_feature_list(self):
        self.data._features_1d = ["b"]
        self.data._features_2d = ["a"]

        self.data._update_feature_list()
        self.assertEqual(self.data.feature_list, ["a", "b"])


    def test_is_adaptive_false(self):
        self.data.U = {"feature1": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)],
                       "directComparison": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)]}

        self.data.features_1d = ["feature1", "directComparison"]

        self.assertFalse(self.data.is_adaptive())


    def test_is_adaptive_true(self):
        self.data.U = {"feature1": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 5)],
                       "directComparison": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)]}

        self.data.features_1d = ["feature1", "directComparison"]

        self.assertTrue(self.data.is_adaptive())

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


    # TODO add this check when changing to python 3
    # def test_loadError(self):
    #     compare_file = "this_file_should_not_exist"
    #
    #     with self.assertRaises(FileNotFoundError):
    #         self.data.load(compare_file)

    def test_save_empty(self):
        data = Data()

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_empty")
        filename = os.path.join(self.output_test_dir, "test_save_empty")

        data.save(filename)

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


    def test_load_empty(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_empty")


        self.data.load(compare_file)

        for data_name in self.data.data_names:
            data = getattr(self.data, data_name)
            self.assertEqual(data, {})

        self.assertEqual(self.data.features_0d, [])
        self.assertEqual(self.data.features_1d, [])
        self.assertEqual(self.data.features_2d, [])
        self.assertEqual(self.data.feature_list, [])
        self.assertEqual(self.data.uncertain_parameters, [])
        self.assertEqual(self.data.xlabel, "")
        self.assertEqual(self.data.ylabel, "")


    def test_remove_only_invalid_results(self):
        self.data.t = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.U = {"feature1": [1., 2.], "directComparison": [3., None]}

        self.data.feature_list = ["directComparison", "feature1"]
        self.data.features_1d = ["directComparison", "feature1"]

        self.data.remove_only_invalid_results()


        self.assertTrue(np.array_equal(self.data.U["feature1"], [1., 2.]))
        self.assertEqual(self.data.U["directComparison"][0], 3.)
        self.assertIsNone(self.data.U["directComparison"][1])
        self.assertTrue(np.array_equal(self.data.t["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.t["directComparison"], [3., 4.]))

        self.assertEqual(self.data.feature_list[0], "directComparison")
        self.assertEqual(self.data.feature_list[1], "feature1")

        self.assertEqual(self.data.features_1d[0], "directComparison")
        self.assertEqual(self.data.features_1d[1], "feature1")


    def test_remove_only_invalid_results_error(self):
        self.data.t = {"feature1": [1., 2.], "directComparison": [3., 4.]}
        self.data.U = {"feature1": [1., 2.], "directComparison": np.array([None, None])}


        self.data.feature_list = ["directComparison", "feature1"]
        self.data.features_1d = ["directComparison", "feature1"]

        self.data.remove_only_invalid_results()

        self.assertTrue(np.array_equal(self.data.U["feature1"], [1., 2.]))
        self.assertEqual(self.data.U["directComparison"],
                         "Only invalid results for all set of parameters")
        self.assertTrue(np.array_equal(self.data.t["feature1"], [1., 2.]))
        self.assertTrue(np.array_equal(self.data.t["directComparison"], [3., 4.]))

        self.assertEqual(self.data.feature_list, ["feature1"])

        self.assertEqual(self.data.features_1d, ["feature1"])


    def test_str(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")

        self.data.load(compare_file)

        # TODO Test that the content of the data string is correct
        self.assertIsInstance(str(self.data), str)



    def test_clear(self):
        self.uncertain_parameters = None

        self.data._features_0d = -1
        self.data._features_1d = -1
        self.data._features_2d = -1
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

        self.data.clear()


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
