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

        self.data_types = ["U", "t", "E", "Var", "p_05", "p_95",
                           "sensitivity_1", "total_sensitivity_1",
                           "sensitivity_t", "total_sensitivity_t", "labels"]


        self.data_information = ["features_0d", "features_1d", "features_2d",
                                 "feature_list", "uncertain_parameters", "model_name"]


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    # def test_features_0d(self):
    #     self.data.features_0d = ["feature0d"]

    #     self.assertEqual(self.data.feature_list, ["feature0d"])
    #     self.assertEqual(self.data.features_0d, ["feature0d"])

    # def test_features_1d(self):
    #     self.data.features_1d = ["feature1d"]

    #     self.assertEqual(self.data.feature_list, ["feature1d"])
    #     self.assertEqual(self.data.features_1d, ["feature1d"])


    def test_add_features(self):
        self.data.add_features("feature1")

        self.assertEqual(self.data.data, {"feature1": {}})

        self.data.add_features(["feature2", "feature3"])

        self.assertEqual(self.data.data, {"feature1": {}, "feature2": {}, "feature3": {}})

    # def test_nan_to_none(self):
    #     a = np.array([0, 1, 2, None, 4, None, None])
    #     b = np.array([0, 1, 2, np.nan, 4, np.nan, np.nan])

    #     result = self.data.nan_to_none(b)

    #     self.assertTrue(np.array_equal(a, result))


    # def test_none_to_nan(self):
    #     a = [0, 1, 2, None, 4, None, None]
    #     b = np.array([0, 1, 2, np.nan, 4, np.nan, np.nan])

    #     result = self.data.none_to_nan(a)

    #     self.assertTrue(np.array_equal(b[~np.isnan(b)], result[~np.isnan(result)]))
    #     self.assertTrue(np.array_equal(np.isnan(b), np.isnan(result)))


    # def test_features_2d(self):
    #     self.data.features_2d = ["feature2d"]

    #     self.assertEqual(self.data.feature_list, ["feature2d"])
    #     self.assertEqual(self.data.features_2d, ["feature2d"])


    # def test_update_feature_list(self):
    #     self.data._features_1d = ["b"]
    #     self.data._features_2d = ["a"]

    #     self.data._update_feature_list()
    #     self.assertEqual(self.data.feature_list, ["a", "b"])


    # def test_is_adaptive_false(self):
    #     self.data.U = {"feature1d": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)],
    #                    "TestingModel1d": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)]}

    #     self.data.features_1d = ["feature1d", "TestingModel1d"]

    #     self.assertFalse(self.data.is_adaptive())


    # def test_is_adaptive_true(self):
    #     self.data.U = {"feature1d": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 5)],
    #                    "TestingModel1d": [np.arange(1, 4), np.arange(1, 4), np.arange(1, 4)]}

    #     self.data.features_1d = ["feature1d", "TestingModel1d"]

    #     self.assertTrue(self.data.is_adaptive())

    def test_save(self):
        self.data.add_features(["feature1d", "TestingModel1d"])

        for data_type in self.data_types:
            self.data["feature1d"][data_type] = [1., 2.]
            self.data["TestingModel1d"][data_type] = [3., 4.]

        self.data["feature1d"]["labels"] = ["xlabel", "ylabel"]
        self.data["TestingModel1d"]["labels"] = ["xlabel", "ylabel"]

        self.data.model_name = "TestingModel1d"
        self.data.uncertain_parameters = ["a", "b"]


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock")
        filename = os.path.join(self.output_test_dir, "test_save_mock")

        self.data.save(filename)

        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)


    # # TODO add this check when changing to python 3
    # # def test_loadError(self):
    # #     compare_file = "this_file_should_not_exist"
    # #
    # #     with self.assertRaises(FileNotFoundError):
    # #         self.data.load(compare_file)


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

        for data_type in self.data_types:
            if data_type == "labels":
                continue
            else:
                self.assertTrue(np.array_equal(self.data["feature1d"][data_type], [1., 2.]))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][data_type], [3., 4.]))

        self.assertEqual(self.data.uncertain_parameters, ["a", "b"])

        self.assertTrue(np.array_equal(self.data["TestingModel1d"]["labels"], ["xlabel", "ylabel"]))
        self.assertTrue(np.array_equal(self.data["feature1d"]["labels"], ["xlabel", "ylabel"]))



    #     self.assertTrue(np.array_equal(self.data.U["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.U["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.E["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.E["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.t["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.t["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.Var["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.Var["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.p_05["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.p_05["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.p_95["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.p_95["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.sensitivity_1["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.sensitivity_1["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.total_sensitivity_1["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.total_sensitivity_1["TestingModel1d"], [3., 4.]))

    #     self.assertTrue(np.array_equal(self.data.sensitivity_t["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.sensitivity_t["TestingModel1d"], [3., 4.]))
    #     self.assertTrue(np.array_equal(self.data.total_sensitivity_t["feature1d"], [1., 2.]))
    #     self.assertTrue(np.array_equal(self.data.total_sensitivity_t["TestingModel1d"], [3., 4.]))


    #     self.assertEqual(self.data.uncertain_parameters[0], "a")
    #     self.assertEqual(self.data.uncertain_parameters[1], "b")

    #     self.assertEqual(self.data.labels["TestingModel1d"], ["xlabel", "ylabel"])
    #     self.assertEqual(self.data.labels["feature1d"], ["xlabel", "ylabel"])

    #     self.assertEqual(self.data.feature_list[0], "TestingModel1d")
    #     self.assertEqual(self.data.feature_list[1], "feature1d")


    def test_load_empty(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_empty")


        self.data.load(compare_file)

        self.assertEqual(self.data.data, {})

        self.assertEqual(self.data.features_0d, [])
        self.assertEqual(self.data.features_1d, [])
        self.assertEqual(self.data.features_2d, [])
        self.assertEqual(self.data.uncertain_parameters, [])
        self.assertEqual(self.data.model_name, "")


    # def test_get_labels(self):
    #     self.data.features_1d = ["model_name", "feature", "feature2"]
    #     self.data.labels = {"model_name": ["x", "y"],
    #                         "feature": ["x", "y"]}
    #     self.data.model_name = "model_name"

    #     self.assertEqual(self.data.get_labels("feature"), ["x", "y"])
    #     self.assertEqual(self.data.get_labels("feature2"), ["x", "y"])

    #     self.data.features_1d = ["model_name", "feature"]
    #     self.data.features_2d = ["feature2"]
    #     self.assertEqual(self.data.get_labels("feature2"), ["", "", ""])

    #     self.data.labels = {"feature": ["x"]}

    #     self.assertEqual(self.data.get_labels("feature2"), ["", "", ""])



    # def test_remove_only_invalid_results(self):
    #     self.data.t = {"feature1d": np.array([1, 2]), "TestingModel1d": np.array([3, 4])}
    #     self.data.U = {"feature1d": np.array([[1, 2], [2, 3]]),
    #                    "TestingModel1d": np.array([[3, 4], [np.nan]])}

    #     self.data.feature_list = ["TestingModel1d", "feature1d"]
    #     self.data.features_1d = ["TestingModel1d", "feature1d"]

    #     self.data.remove_only_invalid_results()

    #     self.assertTrue(np.array_equal(self.data.U["feature1d"], np.array([[1, 2], [2, 3]])))
    #     self.assertTrue(np.array_equal(self.data.t["feature1d"], np.array([1, 2])))
    #     self.assertTrue(np.array_equal(self.data.U["TestingModel1d"], np.array([[3, 4], [np.nan]])))
    #     self.assertTrue(np.array_equal(self.data.t["TestingModel1d"], np.array([3, 4])))

    #     self.assertEqual(self.data.feature_list[0], "TestingModel1d")
    #     self.assertEqual(self.data.feature_list[1], "feature1d")

    #     self.assertEqual(self.data.features_1d[0], "TestingModel1d")
    #     self.assertEqual(self.data.features_1d[1], "feature1d")


    # def test_remove_only_invalid_results_error(self):
    #     self.data.t = {"feature1d": np.array([1, 2]), "TestingModel1d": np.array([3, 4])}
    #     self.data.U = {"feature1d": np.array([[1, 2], [2, 3]]),
    #                    "TestingModel1d": np.array([[np.nan], [np.nan]])}

    #     self.data.feature_list = ["TestingModel1d", "feature1d"]
    #     self.data.features_1d = ["TestingModel1d", "feature1d"]

    #     self.data.remove_only_invalid_results()

    #     self.assertTrue(np.array_equal(self.data.U["feature1d"], np.array([[1, 2], [2, 3]])))
    #     self.assertTrue(np.array_equal(self.data.t["feature1d"], np.array([1, 2])))
    #     self.assertEqual(self.data.U["TestingModel1d"],
    #                      "Only invalid results for all set of parameters")
    #     self.assertTrue(np.array_equal(self.data.t["TestingModel1d"], np.array([3, 4])))


    #     self.assertEqual(self.data.feature_list, ["feature1d"])
    #     self.assertEqual(self.data.features_1d, ["feature1d"])


    # def test_str(self):
    #     folder = os.path.dirname(os.path.realpath(__file__))
    #     compare_file = os.path.join(folder, "data/TestingModel1d.h5")

    #     self.data.load(compare_file)

    #     # TODO Test that the content of the data string is correct
    #     self.assertIsInstance(str(self.data), str)



    def test_clear(self):
        self.uncertain_parameters = None

        self.data.features_0d = -1
        self.data.features_1d = -1
        self.data.features_2d = -1
        self.data.data = -1

        self.data.clear()


        self.assertEqual(self.data.features_0d, [])
        self.assertEqual(self.data.features_1d, [])
        self.assertEqual(self.data.features_2d, [])
        self.assertEqual(self.data.data, {})

