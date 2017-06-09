import os

import numpy as np
import uncertainpy as un
from testing_classes import TestingModel1d, TestingModel0d, TestingModel2d
from testing_classes import TestingFeatures, model_function

folder = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(folder, "data")

seed = 10


def generate_data_polynomial_chaos():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)


    test.polynomial_chaos()


def generate_data_PC_model_function():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)


    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    test = un.UncertaintyEstimation(model_function,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)


    test.polynomial_chaos()



def generate_data_PC_0D():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel0d()

    features = TestingFeatures(features_to_run=None)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)


    test.polynomial_chaos()


def generate_data_PC_2D():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel2d()

    features = TestingFeatures(features_to_run=None)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)


    test.polynomial_chaos()


def generate_data_PC_rosenblatt():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)


    test.polynomial_chaos(rosenblatt=True, filename="TestingModel1d_Rosenblatt")





def generate_data_polynomial_chaos_single():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)



    test.polynomial_chaos_single()


def generate_data_monte_carlo():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)

    test.monte_carlo(filename="TestingModel1d_MC",)


def generate_data_monte_carlo_single():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    parameters=parameters,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error",
                                    seed=seed,
                                    nr_mc_samples=10)


    test.monte_carlo_single(filename="TestingModel1d_MC")



# def generate_data_comparemonte_carlo():  # pragma: no cover
#     parameterlist = [["a", 1, None],
#                      ["b", 2, None]]
#
#     parameters = un.Parameters(parameterlist)
#     model = TestingModel1d(parameters)
#     model.set_all_distributions(un.Distribution(0.5).uniform)
#
#
#     uncertainty = un.UncertaintyEstimations(model,
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

    data.uncertain_parameters = ["a", "b"]
    data.feature_list = ["TestingModel1d", "feature1d"]
    data.t = {"feature1d": [1., 2.], "TestingModel1d": [3., 4.]}
    data.U = {"feature1d": [1., 2.], "TestingModel1d": [3., 4.]}
    data.E = {"feature1d": [1., 2.], "TestingModel1d": [3., 4.]}
    data.Var = {"feature1d": [1., 2.], "TestingModel1d": [3., 4.]}
    data.p_05 = {"feature1d": [1., 2.], "TestingModel1d": [3., 4.]}
    data.p_95 = {"feature1d": [1., 2.], "TestingModel1d": [3., 4.]}
    data.sensitivity_1 = {"feature1d": [1, 2], "TestingModel1d": [3., 4.]}
    data.total_sensitivity_1 = {"feature1d": [1, 2], "TestingModel1d": [3., 4.]}
    data.sensitivity_t = {"feature1d": [1, 2], "TestingModel1d": [3., 4.]}
    data.total_sensitivity_t = {"feature1d": [1, 2], "TestingModel1d": [3., 4.]}

    data.model_name = "TestingModel1d"
    data.labels = {"feature1d": ["xlabel", "ylabel"],
                   "TestingModel1d": ["xlabel", "ylabel"]}

    data.features_0d = ["TestingModel1d", "feature1d"]
    data.features_1d = []
    data.features_2d = []


    data.save(os.path.join(test_data_dir, "test_save_mock"))


def generate_data_empty():  # pragma: no cover
    data = un.Data()

    data.save(os.path.join(test_data_dir, "test_save_empty"))


def generate_data_uncertainty_calculations():  # pragma: no cover
    np.random.seed(seed)

    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.set_all_distributions(un.Distribution(0.5).uniform)

    model = TestingModel1d()

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    uncertainty_calculations = un.UncertaintyCalculations(model=model,
                                                          features=features,
                                                          parameters=parameters,
                                                          seed=seed,
                                                          nr_mc_samples=10)


    data = uncertainty_calculations.polynomial_chaos("a")
    data.save(os.path.join(test_data_dir, "UncertaintyCalculations_single-parameter-a.h5"))

    np.random.seed(seed)

    data = uncertainty_calculations.polynomial_chaos("b")
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
