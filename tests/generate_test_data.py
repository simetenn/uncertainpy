import os

import numpy as np
import uncertainpy as un
from testing_classes import TestingModel1d, TestingModel0d
from testing_classes import TestingFeatures

folder = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(folder, "data")

seed = 10


def generate_data_PC():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.setAllDistributions(un.Distribution(0.5).uniform)

    model = TestingModel1d(parameters)

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    uncertainty_calculations = un.UncertaintyCalculations(seed=seed, nr_mc_samples=10)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    uncertainty_calculations=uncertainty_calculations,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error")


    test.PC()



def generate_data_PC0D():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.setAllDistributions(un.Distribution(0.5).uniform)

    model = TestingModel0d(parameters)

    features = TestingFeatures(features_to_run=None)

    uncertainty_calculations = un.UncertaintyCalculations(seed=seed, nr_mc_samples=10)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    uncertainty_calculations=uncertainty_calculations,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error")


    test.PC()


def generate_data_PCRosenblatt():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.setAllDistributions(un.Distribution(0.5).uniform)

    model = TestingModel1d(parameters)

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    uncertainty_calculations = un.UncertaintyCalculations(seed=seed, nr_mc_samples=10)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    uncertainty_calculations=uncertainty_calculations,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    output_data_filename="TestingModel1d_Rosenblatt",
                                    verbose_level="error")


    test.PC(rosenblatt=True)



def generate_data_PCSingle():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.setAllDistributions(un.Distribution(0.5).uniform)

    model = TestingModel1d(parameters)

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    uncertainty_calculations = un.UncertaintyCalculations(seed=seed, nr_mc_samples=10)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    uncertainty_calculations=uncertainty_calculations,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    verbose_level="error")



    test.PCSingle()


def generate_data_MC():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.setAllDistributions(un.Distribution(0.5).uniform)

    model = TestingModel1d(parameters)

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    uncertainty_calculations = un.UncertaintyCalculations(seed=seed, nr_mc_samples=10)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    uncertainty_calculations=uncertainty_calculations,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    output_data_filename="TestingModel1d_MC",
                                    verbose_level="error")

    test.MC()


def generate_data_MCSingle():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.setAllDistributions(un.Distribution(0.5).uniform)

    model = TestingModel1d(parameters)

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])

    uncertainty_calculations = un.UncertaintyCalculations(seed=seed, nr_mc_samples=10)


    test = un.UncertaintyEstimation(model,
                                    features=features,
                                    uncertainty_calculations=uncertainty_calculations,
                                    output_dir_data=test_data_dir,
                                    save_figures=False,
                                    output_data_filename="TestingModel1d_MC",
                                    verbose_level="error")


    test.MCSingle()



def generate_data_compareMC():  # pragma: no cover
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    model = TestingModel1d(parameters)
    model.setAllDistributions(un.Distribution(0.5).uniform)


    uncertainty = un.UncertaintyEstimations(model,
                                            features=TestingFeatures(),
                                            verbose_level="error",
                                            output_dir_data=test_data_dir,
                                            save_figures=False,
                                            nr_mc_samples=10**1,
                                            seed=seed)


    mc_samples = [10, 100]
    uncertainty.compareMC(mc_samples)



def generate_data_data():  # pragma: no cover
    data = un.Data()

    data.uncertain_parameters = ["a", "b"]
    data.feature_list = ["directComparison", "feature1"]
    data.t = {"feature1": [1., 2.], "directComparison": [3., 4.]}
    data.U = {"feature1": [1., 2.], "directComparison": [3., 4.]}
    data.E = {"feature1": [1., 2.], "directComparison": [3., 4.]}
    data.Var = {"feature1": [1., 2.], "directComparison": [3., 4.]}
    data.p_05 = {"feature1": [1., 2.], "directComparison": [3., 4.]}
    data.p_95 = {"feature1": [1., 2.], "directComparison": [3., 4.]}
    data.sensitivity_1 = {"feature1": [1, 2], "directComparison": [3., 4.]}
    data.total_sensitivity_1 = {"feature1": [1, 2], "directComparison": [3., 4.]}
    data.sensitivity_t = {"feature1": [1, 2], "directComparison": [3., 4.]}
    data.total_sensitivity_t = {"feature1": [1, 2], "directComparison": [3., 4.]}
    data.xlabel = "xlabel"
    data.ylabel = "ylabel"

    data.features_0d = ["directComparison", "feature1"]
    data.features_1d = []
    data.features_2d = []


    data.save(os.path.join(test_data_dir, "test_save_mock"))



def generate_data_UncertaintyCalculations():  # pragma: no cover
    np.random.seed(seed)

    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = un.Parameters(parameterlist)
    parameters.setAllDistributions(un.Distribution(0.5).uniform)

    model = TestingModel1d(parameters)

    features = TestingFeatures(features_to_run=["feature0d",
                                                "feature1d",
                                                "feature2d"])


    uncertainty_calculations = un.UncertaintyCalculations(model,
                                                          features,
                                                          seed=seed,
                                                          nr_mc_samples=10)


    data = uncertainty_calculations.PC("a")
    data.save(os.path.join(test_data_dir, "UncertaintyCalculations_single-parameter-a.h5"))

    np.random.seed(seed)

    data = uncertainty_calculations.PC("b")
    data.save(os.path.join(test_data_dir, "UncertaintyCalculations_single-parameter-b.h5"))



if __name__ == "__main__":  # pragma: no cover
    generate_data_PC()
    generate_data_PC0D()
    generate_data_PCSingle()

    generate_data_MC()
    generate_data_MCSingle()
    # generate_data_compareMC()
    generate_data_PCRosenblatt()
    generate_data_UncertaintyCalculations()
    generate_data_data()
