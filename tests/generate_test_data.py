import os

import numpy as np
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

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    test = un.UncertaintyQuantification(model,
                                    features=features,
                                    parameters=parameters,
                                    verbose_level="error")


    test.polynomial_chaos(output_dir_data=test_data_dir,
                          seed=seed,
                          save_figures=False)


def generate_data_PC_model_function():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    test = un.UncertaintyQuantification(model_function,
                                        features=features,
                                        parameters=parameters,
                                        verbose_level="error")

    test.polynomial_chaos(output_dir_data=test_data_dir,
                          seed=seed,
                          save_figures=False)



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
                                        verbose_level="error")


    test.polynomial_chaos(output_dir_data=test_data_dir,
                          seed=seed,
                          save_figures=False)


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
                                        verbose_level="error")


    test.polynomial_chaos(output_dir_data=test_data_dir,
                          seed=seed,
                          save_figures=False)


def generate_data_PC_rosenblatt():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        verbose_level="error")


    test.polynomial_chaos(rosenblatt=True,
                          filename="TestingModel1d_Rosenblatt",
                          output_dir_data=test_data_dir,
                          save_figures=False)





def generate_data_polynomial_chaos_single():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        verbose_level="error")



    test.polynomial_chaos_single(output_dir_data=test_data_dir,
                                 seed=seed,
                                 save_figures=False)


def generate_data_monte_carlo():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        verbose_level="error")

    test.monte_carlo(filename="TestingModel1d_MC",
                     output_dir_data=test_data_dir,
                     seed=seed,
                     nr_samples=10,
                     save_figures=False)


def generate_data_monte_carlo_single():  # pragma: no cover
    parameter_list = [["a", 1, None],
                      ["b", 2, None]]

    parameters = un.Parameters(parameter_list)
    parameters.set_all_distributions(un.uniform(0.5))

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    test = un.UncertaintyQuantification(model,
                                        features=features,
                                        parameters=parameters,
                                        verbose_level="error")


    test.monte_carlo_single(filename="TestingModel1d_MC",
                            output_dir_data=test_data_dir,
                            seed=seed,
                            nr_samples=10,
                            save_figures=False)



# def generate_data_comparemonte_carlo():  # pragma: no cover
#     parameter_list = [["a", 1, None],
#                      ["b", 2, None]]
#
#     parameters = un.Parameters(parameter_list)
#     model = TestingModel1d(parameters)
#     model.set_all_distributions(un.uniform(0.5))
#
#
#     uncertainty = un.UncertaintyQuantifications(model,
#                                             features=TestingFeatures(),
#                                             verbose_level="error",
#                                             output_dir_data=test_data_dir,
#                                             save_figures=False,
#                                             nr_mc_samples=10**1,
#                                             seed=seed)
#
#
#     mc_samples = [10, 100]
#     uncertainty.comparemonte_carlo(mc_samples)



def generate_data_data():  # pragma: no cover
    data = un.Data()
    data_types = ["values", "time", "mean", "variance", "percentile_5", "percentile_95",
                  "sensitivity_1", "sensitivity_1_sum",
                  "sensitivity_t", "sensitivity_t_sum"]

    data.add_features(["feature1d", "TestingModel1d"])

    for data_type in data_types:
        data["feature1d"][data_type] = [1., 2.]
        data["TestingModel1d"][data_type] = [3., 4.]

    data["feature1d"]["labels"] = ["xlabel", "ylabel"]
    data["TestingModel1d"]["labels"] = ["xlabel", "ylabel"]

    data.uncertain_parameters = ["a", "b"]
    data.model_name = "TestingModel1d"

    data.save(os.path.join(test_data_dir, "test_save_mock"))


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

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    uncertainty_calculations = un.core.UncertaintyCalculations(model=model,
                                                               features=features,
                                                               parameters=parameters)


    data = uncertainty_calculations.polynomial_chaos("a", seed=seed)
    data.save(os.path.join(test_data_dir, "UncertaintyCalculations_single-parameter-a.h5"))

    np.random.seed(seed)

    data = uncertainty_calculations.polynomial_chaos("b", seed=seed)
    data.save(os.path.join(test_data_dir, "UncertaintyCalculations_single-parameter-b.h5"))



if __name__ == "__main__":  # pragma: no cover
    generate_data_polynomial_chaos()
    generate_data_PC_model_function()
    generate_data_PC_0D()
    generate_data_PC_2D()
    generate_data_polynomial_chaos_single()

    generate_data_monte_carlo()
    generate_data_monte_carlo_single()
    # generate_data_comparemonte_carlo()
    generate_data_PC_rosenblatt()
    generate_data_uncertainty_calculations()
    generate_data_data()
    generate_data_empty()
