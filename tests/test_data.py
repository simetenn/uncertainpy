import unittest
import shutil
import os
import subprocess

import numpy as np

from uncertainpy import Data
from uncertainpy.data import DataFeature


class TestDataFeature(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


        self.data_feature = DataFeature("test")

        self.statistical_metrics = ["evaluations", "time", "mean", "variance",
                                    "percentile_5", "percentile_95",
                                    "sobol_first", "sobol_first_average",
                                    "sobol_total", "sobol_total_average"]


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)



    def test_getitem(self):
        for statistical_metric in self.statistical_metrics:
            self.assertIsNone(self.data_feature[statistical_metric])

        self.assertEqual(self.data_feature["labels"], [])


    def test_getitem_error(self):
        with self.assertRaises(AttributeError):
            self.data_feature["error"]


    def test_setitem(self):
        for statistical_metric in self.statistical_metrics:
            self.data_feature[statistical_metric] = 2

        for statistical_metric in self.statistical_metrics:
            self.assertEqual(self.data_feature[statistical_metric], 2)


    def test_get_metrics(self):
        for statistical_metric in self.statistical_metrics:
            self.data_feature[statistical_metric] = 2

        self.assertEqual(set(self.statistical_metrics), set(self.data_feature.get_metrics()))

        self.data_feature.x = 2

        self.assertEqual(set(self.statistical_metrics + ["x"]), set(self.data_feature.get_metrics()))


    def test_delitem(self):
        for statistical_metric in self.statistical_metrics:
            self.data_feature[statistical_metric] = 2

        for statistical_metric in self.statistical_metrics:
            del self.data_feature[statistical_metric]

        self.assertEqual(self.data_feature.get_metrics(), [])

    def test_str(self):
        for statistical_metric in self.statistical_metrics:
            self.data_feature[statistical_metric] = [2, 2, 2]

        # TODO Test that the content of the data string is correct
        self.assertIsInstance(str(self.data_feature), str)


    def test_iter(self):
        for statistical_metric in self.statistical_metrics:
            self.data_feature[statistical_metric] = 2

        result = []
        for statistical_metric in self.data_feature:
            result.append(statistical_metric)

        self.assertEqual(set(self.data_feature.get_metrics()), set(self.statistical_metrics))


    def test_len(self):
        for statistical_metric in self.statistical_metrics:
            self.data_feature[statistical_metric] = 2

        self.assertEqual(len(self.data_feature), len(self.statistical_metrics))


    def test_ndim(self):
        self.data_feature.evaluations = [[[1, 2, 3], [1, 2, 3]]]
        self.assertEqual(self.data_feature.ndim(), 2)

        self.data_feature.evaluations = [1]
        self.assertEqual(self.data_feature.ndim(), 0)

        self.data_feature.evaluations = [np.arange(0, 10)]
        self.assertEqual(self.data_feature.ndim(), 1)

        self.data_feature.evaluations =[np.array([np.arange(0, 10),
                                                  np.arange(0, 10)])]

        self.assertEqual(self.data_feature.ndim(), 2)

        self.data_feature.evaluations = [[]]
        self.assertEqual(self.data_feature.ndim(), 1)


        self.data_feature.evaluations = None
        self.assertIsNone(self.data_feature.ndim())

        self.data_feature.evaluations = [np.nan, np.nan]
        self.assertIsNone(self.data_feature.ndim())


    def test_contains(self):
        self.assertFalse("error" in self.data_feature)

        self.data_feature.evaluations = 2

        self.assertTrue("evaluations" in self.data_feature)




class TestData(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


        self.data = Data(logger_level="error")

        self.statistical_metrics = ["evaluations", "time", "mean", "variance", "percentile_5", "percentile_95",
                                    "sobol_first", "sobol_first_average",
                                    "sobol_total", "sobol_total_average"]


        self.data_information = ["uncertain_parameters", "model_name",
                                 "incomplete", "method", "version"]


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_add_features(self):
        self.data.add_features("feature1")

        self.assertEqual(self.data.data, {"feature1": DataFeature("feature1")})

        self.data.add_features(u"feature1")

        self.assertEqual(self.data.data, {"feature1": DataFeature("feature1")})

        self.data.add_features(["feature2", "feature3"])

        self.assertEqual(self.data.data, {"feature1": DataFeature("feature1"),
                                          "feature2": DataFeature("feature2"),
                                          "feature3": DataFeature("feature3")})


    def test_save(self):
        self.setup_mock_data(self.data)

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock")
        filename = os.path.join(self.output_test_dir, "test_save_mock")

        self.data.save(filename)

        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)

    def test_backend_error(self):
        with self.assertRaises(ValueError):
            self.data = Data(backend="not a backend")


    def test_save_load_exdir(self):
        self.data = Data(logger_level="error", backend="exdir")

        self.setup_mock_data(self.data)


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock")
        filename = os.path.join(self.output_test_dir, "test_save_mock")

        self.data.save(filename)

        new_data = Data(filename, backend="exdir")

        for statistical_metric in self.statistical_metrics:
            new_data["feature1d"][statistical_metric] = np.array([1., 2.])
            new_data["TestingModel1d"][statistical_metric] = np.array([3., 4.])

        new_data["feature1d"]["labels"] = ["xlabel", "ylabel"]
        new_data["TestingModel1d"]["labels"] = ["xlabel", "ylabel"]

        new_data.model_name = "TestingModel1d"
        new_data.uncertain_parameters = ["a", "b"]
        new_data.method = "mock"
        new_data.seed = 10
        new_data.incomplete = ["a", "b"]
        new_data.error = ["feature1d"]


    def setup_mock_data(self, data):
        data.add_features(["feature1d", "TestingModel1d"])

        for statistical_metric in self.statistical_metrics:
            data["feature1d"][statistical_metric] = np.array([1., 2.])
            data["TestingModel1d"][statistical_metric] = np.array([3., 4.])

        data["feature1d"]["labels"] = ["xlabel", "ylabel"]
        data["TestingModel1d"]["labels"] = ["xlabel", "ylabel"]

        data.model_name = "TestingModel1d"
        data.uncertain_parameters = ["a", "b"]
        data.method = "mock"
        data.seed = 10
        data.incomplete = ["a", "b"]
        data.error = ["feature1d"]

        return data


    def test_save_irregular(self):
        self.data.add_features(["feature1d", "TestingModel1d"])

        self.setup_mock_data(self.data)

        self.data.model_ignore = True
        self.data["TestingModel1d"].evaluations = [[1, 2], [np.nan], [1, [2, 3], 3], [1], 3, [3, 4, 5], [1, 2], [], [3, 4, 5], [], [3, 4, 5]]
        self.data["TestingModel1d"].time = [[1, 2], [np.nan], [1, [2, 3], 3], [1], 3, [3, 4, 5], [1, 2], [], [3, 4, 5], [], [3, 4, 5]]


        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock_irregular")
        filename = os.path.join(self.output_test_dir, "test_save_mock_irregular")

        self.data.save(filename)

        result = subprocess.call(["h5diff", filename, compare_file])

        self.assertEqual(result, 0)


    def test_load_irregular(self):

        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock_irregular")

        self.data.load(compare_file)

        for statistical_metric in self.statistical_metrics:
            if statistical_metric in ["evaluations", "time"]:
                self.assertTrue(np.array_equal(self.data["feature1d"][statistical_metric], [1., 2.]))

                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][0], [1., 2.]))
                self.assertTrue(np.isnan(self.data["TestingModel1d"][statistical_metric][1]))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][2][0], 1))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][2][1], np.array([2, 3])))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][2][2], 3))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][3], np.array([1])))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][4], 3))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][5], [3, 4, 5]))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][6], [1., 2.]))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][7], np.array([])))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][8], [3, 4, 5]))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][9], np.array([])))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric][10], [3, 4, 5]))

            else:
                self.assertTrue(np.array_equal(self.data["feature1d"][statistical_metric], [1., 2.]))
                self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric], [3., 4.]))

        self.assertEqual(self.data.uncertain_parameters, ["a", "b"])
        self.assertEqual(self.data.incomplete, ["a", "b"])
        self.assertEqual(self.data.error, ["feature1d"])

        self.assertEqual(self.data.model_name, "TestingModel1d")
        self.assertEqual(self.data.method, "mock")
        self.assertEqual(self.data.seed, 10)

        self.assertTrue(np.array_equal(self.data["TestingModel1d"]["labels"], ["xlabel", "ylabel"]))
        self.assertTrue(np.array_equal(self.data["feature1d"]["labels"], ["xlabel", "ylabel"]))




    # # TODO add this check when changing to python 3
    # # def test_loadError(self):
    # #     compare_file = "this_file_should_not_exist"
    # #
    # #     with self.assertRaises(FileNotFoundError):
    # #         self.data.load(compare_file)

    def test_seed(self):
        data = Data()
        self.assertEqual(data.seed, "")

        data.seed = 10
        self.assertEqual(data.seed, 10)

        data.seed = None
        self.assertEqual(data.seed, "")


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

        for statistical_metric in self.statistical_metrics:
            self.assertTrue(np.array_equal(self.data["feature1d"][statistical_metric], [1., 2.]))
            self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric], [3., 4.]))

        self.assertEqual(self.data.uncertain_parameters, ["a", "b"])
        self.assertEqual(self.data.incomplete, ["a", "b"])
        self.assertEqual(self.data.error, ["feature1d"])

        self.assertEqual(self.data.model_name, "TestingModel1d")
        self.assertEqual(self.data.method, "mock")
        self.assertEqual(self.data.seed, 10)
        self.assertTrue(np.array_equal(self.data["TestingModel1d"]["labels"], ["xlabel", "ylabel"]))
        self.assertTrue(np.array_equal(self.data["feature1d"]["labels"], ["xlabel", "ylabel"]))



    def test_load_missing(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_mock_missing")

        self.data.load(compare_file)

        for statistical_metric in self.statistical_metrics:
            self.assertTrue(np.array_equal(self.data["feature1d"][statistical_metric], [1., 2.]))
            self.assertTrue(np.array_equal(self.data["TestingModel1d"][statistical_metric], [3., 4.]))

        self.assertEqual(self.data.uncertain_parameters, ["a", "b"])
        self.assertEqual(self.data.incomplete, [])
        self.assertEqual(self.data.error, ["feature1d"])

        self.assertEqual(self.data.model_name, "TestingModel1d")
        self.assertEqual(self.data.method, "mock")
        self.assertEqual(self.data.seed, "")

        self.assertTrue(np.array_equal(self.data["TestingModel1d"]["labels"], ["xlabel", "ylabel"]))
        self.assertTrue(np.array_equal(self.data["feature1d"]["labels"], ["xlabel", "ylabel"]))



    def test_load_empty(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/test_save_empty")


        self.data.load(compare_file)

        self.assertEqual(self.data.data, {})

        self.assertEqual(self.data.uncertain_parameters, [])
        self.assertEqual(self.data.model_name, "")
        self.assertEqual(self.data.uncertain_parameters, [])
        self.assertEqual(self.data.incomplete, [])
        self.assertEqual(self.data.method, "")
        self.assertEqual(self.data.seed, "")


    def test_get_labels(self):
        self.data.add_features(["model_name", "feature", "feature2"])

        self.data["model_name"].labels = ["x", "y"]
        self.data["feature"].labels = ["x", "y"]
        self.data["model_name"].evaluations = [[1, 2], [1, 2]]
        self.data["feature"].evaluations = [[1, 2], [1, 2]]
        self.data["feature2"].evaluations = [[1, 2], [1, 2]]

        self.data.model_name = "model_name"

        self.assertEqual(self.data.get_labels("feature"), ["x", "y"])
        self.assertEqual(self.data.get_labels("feature2"), ["x", "y"])

        self.data["feature2"].evaluations = [[[1], [2]], [[1], [2]]]
        self.assertEqual(self.data.get_labels("feature2"), ["", "", ""])

        self.data["feature"]["labels"] = ["x"]

        self.assertEqual(self.data.get_labels("feature"), ["x"])



    def test_getitem(self):
        self.data.data["test1"] = 1
        self.data.data["test2"] = 2

        self.assertEqual(self.data["test1"], 1)
        self.assertEqual(self.data["test2"], 2)



    def test_iter(self):
        self.data.data["test1"] = 1
        self.data.data["test2"] = 2

        result = []
        for feature in self.data:
            result.append(feature)

        self.assertEqual(result, ["test1", "test2"])



    def test_iter_error(self):
        self.data.data["test1"] = 1
        self.data.data["test2"] = 2

        self.data.error = ["test2"]
        result = []
        for feature in self.data:
            result.append(feature)

        self.assertEqual(result, ["test1"])


    def test_iter_error_all(self):
        self.data.data["test1"] = 1
        self.data.data["test2"] = 2

        self.data.error = ["test1", "test2"]
        result = []
        for feature in self.data:
            result.append(feature)

        self.assertEqual(result, [])


    def test_len(self):
        self.data.data["test1"] = 1
        self.data.data["test2"] = 2

        self.data.error = ["test2"]

        self.assertEqual(len(self.data), 1)


    def test_delitem(self):
        self.data.data["test1"] = 1
        self.data.data["test2"] = 2

        del self.data["test2"]

        self.assertTrue("test1" in self.data)
        self.assertFalse("test2" in self.data)


    def test_remove_only_invalid_features(self):
        self.data.add_features(["feature1d", "TestingModel1d"])
        self.data["feature1d"]["evaluations"] = np.array([[1, 2], [2, 3]])
        self.data["TestingModel1d"]["evaluations"] = np.array([[3, 4], [np.nan]])

        self.data["feature1d"]["time"] = np.array([1, 2])
        self.data["TestingModel1d"]["time"] = np.array([3, 4])

        self.data.remove_only_invalid_features()

        self.assertTrue(np.array_equal(self.data["feature1d"]["evaluations"], np.array([[1, 2], [2, 3]])))
        self.assertTrue(np.array_equal(self.data["feature1d"]["time"], np.array([1, 2])))
        self.assertTrue(np.array_equal(self.data["TestingModel1d"]["evaluations"],
                                       np.array([[3, 4], [np.nan]])))
        self.assertTrue(np.array_equal(self.data["TestingModel1d"]["time"], np.array([3, 4])))



    def test_remove_only_invalid_features_error(self):
        self.data.add_features(["feature1d", "TestingModel1d"])
        self.data["feature1d"]["evaluations"] = np.array([[1, 2], [2, 3]])
        self.data["TestingModel1d"]["evaluations"] = np.array([[np.nan], [np.nan]])

        self.data["feature1d"]["time"] = np.array([1, 2])
        self.data["TestingModel1d"]["time"] = np.array([3, 4])

        self.data.remove_only_invalid_features()

        self.assertTrue(np.array_equal(self.data["feature1d"]["evaluations"], np.array([[1, 2], [2, 3]])))
        self.assertTrue(np.array_equal(self.data["feature1d"]["time"], np.array([1, 2])))
        self.assertFalse("TestingModel1d" in self.data)



    def test_str(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/TestingModel1d.h5")

        self.data.load(compare_file)

        # TODO Test that the content of the data string is correct
        self.assertIsInstance(str(self.data), str)



    def test_clear(self):
        self.data.uncertain_parameters = -1
        self.data.model_name = -1
        self.data.data = -1
        self.data.incomplete = -1
        self.data.method = -1
        self.data.seed = -1

        self.data.clear()

        self.assertEqual(self.data.model_name, "")
        self.assertEqual(self.data.data, {})
        self.assertEqual(self.data.uncertain_parameters, [])
        self.assertEqual(self.data.incomplete, [])
        self.assertEqual(self.data.model_name, "")
        self.assertEqual(self.data.method, "")
        self.assertEqual(self.data.seed, "")


    def test_ndim(self):

        self.data.add_features(["feature0d", "feature1d", "feature2d", "feature_invalid", "empty", "test"])

        self.data["feature0d"].evaluations = [1]
        self.data["feature1d"].evaluations = [np.arange(0, 10)]
        self.data["feature2d"].evaluations = [np.array([np.arange(0, 10),
                                              np.arange(0, 10)])]
        self.data["feature_invalid"].evaluations = [np.nan]
        self.data["test"].evaluations = [np.nan, np.arange(0, 10)]

        self.assertEqual(self.data.ndim("feature0d"), 0)
        self.assertEqual(self.data.ndim("feature1d"), 1)
        self.assertEqual(self.data.ndim("feature2d"), 2)
        self.assertIsNone(self.data.ndim("feature_invalid"))
        self.assertIsNone(self.data.ndim("empty"))