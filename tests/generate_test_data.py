from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import h5py
import uncertainpy as un
from testing_classes import TestingModel1d, TestingModel0d, TestingModel2d
from testing_classes import TestingFeatures, model_function

folder = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(folder, "data")

seed = 10


def generate_data_polynomial_chaos():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])


    test = un.UncertaintyQuantification(model,
                                    features=features,
                                    parameters=parameters,
                                    logger_level="error",
                                    logger_filename=None)


    test.polynomial_chaos(data_folder=test_data_dir,
                          seed=seed,
                          plot=None)


def generate_data_PC_spectral():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)


    test.polynomial_chaos(method="spectral",
                          filename="TestingModel1d_spectral",
                          seed=seed,
                          data_folder=test_data_dir,
                          plot=None)


def generate_data_PC_model_function():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])

    test = un.UncertaintyQuantification(model_function,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)

    test.polynomial_chaos(data_folder=test_data_dir,
                          seed=seed,
                          plot=None)



def generate_data_PC_0D():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel0d()

    features = TestingFeatures(features_to_run=None)


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)


    test.polynomial_chaos(data_folder=test_data_dir,
                          seed=seed,
                          plot=None)


def generate_data_PC_2D():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel2d()

    features = TestingFeatures(features_to_run=None)


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)


    test.polynomial_chaos(data_folder=test_data_dir,
                          seed=seed,
                          plot=None)


def generate_data_PC_rosenblatt():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)


    test.polynomial_chaos(rosenblatt=True,
                          filename="TestingModel1d_Rosenblatt",
                          data_folder=test_data_dir,
                          plot=None,
                          seed=seed)



def generate_data_PC_rosenblatt_spectral():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)


    test.polynomial_chaos(rosenblatt=True,
                          method="spectral",
                          filename="TestingModel1d_Rosenblatt_spectral",
                          data_folder=test_data_dir,
                          plot=None,
                          seed=seed)



def generate_data_polynomial_chaos_single():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])

    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)



    test.polynomial_chaos_single(data_folder=test_data_dir,
                                 seed=seed,
                                 plot=None)


def generate_data_monte_carlo():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])

    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)

    test.monte_carlo(filename="TestingModel1d_MC",
                     data_folder=test_data_dir,
                     seed=seed,
                     nr_samples=10,
                     plot=None)


def generate_data_monte_carlo_single():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        logger_level="error",
                                        logger_filename=None)


    test.monte_carlo_single(filename="TestingModel1d_MC",
                            data_folder=test_data_dir,
                            seed=seed,
                            nr_samples=10,
                            plot=None)


def setup_mock_data():  # pragma: no cover
    data = un.Data()
    data_types = ["evaluations", "time", "mean", "variance", "percentile_5", "percentile_95",
                  "sobol_first", "sobol_first_average",
                  "sobol_total", "sobol_total_average"]

    data.add_features(["feature1d", "TestingModel1d"])

    for data_type in data_types:
        data["feature1d"][data_type] = np.array([1., 2.])
        data["TestingModel1d"][data_type] = np.array([3., 4.])

    data["feature1d"]["labels"] = ["xlabel", "ylabel"]
    data["TestingModel1d"]["labels"] = ["xlabel", "ylabel"]

    data.uncertain_parameters = ["a", "b"]
    data.model_name = "TestingModel1d"
    data.method = "mock"
    data.seed = 10
    data.incomplete = ["a", "b"]
    data.error = ["feature1d"]

    return data


def generate_data_data():  # pragma: no cover
    data = setup_mock_data()

    data.save(os.path.join(test_data_dir, "test_save_mock"))



def generate_data_data_missing():  # pragma: no cover
    data = setup_mock_data()

    data.save(os.path.join(test_data_dir, "test_save_mock_missing"))

    with h5py.File(os.path.join(test_data_dir, "test_save_mock_missing"), "a") as f:
        del f.attrs["incomplete results"]
        del f.attrs["seed"]


def generate_data_data_irregular():  # pragma: no cover
    data = un.Data()
    data_types = ["evaluations", "time", "mean", "variance", "percentile_5", "percentile_95",
                  "sobol_first", "sobol_first_average",
                  "sobol_total", "sobol_total_average"]

    data.add_features(["feature1d", "TestingModel1d"])

    for data_type in data_types:
        data["feature1d"][data_type] = [1., 2.]
        data["TestingModel1d"][data_type] = [3., 4.]


    data["TestingModel1d"].evaluations = [[1, 2], [np.nan], [1, [2, 3], 3], [1], 3, [3, 4, 5], [1, 2], [], [3, 4, 5], [], [3, 4, 5]]
    data["TestingModel1d"].time = [[1, 2], [np.nan], [1, [2, 3], 3], [1], 3, [3, 4, 5], [1, 2], [], [3, 4, 5], [], [3, 4, 5]]

    data["feature1d"]["labels"] = ["xlabel", "ylabel"]
    data["TestingModel1d"]["labels"] = ["xlabel", "ylabel"]

    data.uncertain_parameters = ["a", "b"]
    data.model_name = "TestingModel1d"
    data.method = "mock"
    data.seed = 10
    data.incomplete = ["a", "b"]
    data.error = ["feature1d"]
    data.model_ignore = True

    data.save(os.path.join(test_data_dir, "test_save_mock_irregular"))


def generate_data_empty():  # pragma: no cover
    data = un.Data()

    data.save(os.path.join(test_data_dir, "test_save_empty"))


def generate_data_uncertainty_calculations():  # pragma: no cover
    np.random.seed(seed)

    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d_var",
                                                "feature1d_var",
                                                "feature2d_var"])

    uncertainty_calculations = un.core.UncertaintyCalculations(model=model,
                                                               features=features,
                                                               parameters=parameters)


    data = uncertainty_calculations.polynomial_chaos(uncertain_parameters="a", seed=seed)
    data.save(os.path.join(test_data_dir, "UncertaintyCalculations_a.h5"))

    np.random.seed(seed)

    data = uncertainty_calculations.polynomial_chaos(uncertain_parameters="b", seed=seed)
    data.save(os.path.join(test_data_dir, "UncertaintyCalculations_b.h5"))



if __name__ == "__main__":  # pragma: no cover
    generate_data_polynomial_chaos()
    generate_data_PC_model_function()
    generate_data_PC_0D()
    generate_data_PC_2D()
    generate_data_polynomial_chaos_single()
    generate_data_PC_spectral()
    generate_data_PC_rosenblatt_spectral()

    generate_data_monte_carlo()
    generate_data_monte_carlo_single()
    generate_data_PC_rosenblatt()
    generate_data_uncertainty_calculations()
    generate_data_data()
    generate_data_empty()
    generate_data_data_irregular()
    generate_data_data_missing()
