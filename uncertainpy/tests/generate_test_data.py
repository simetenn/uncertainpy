import os
import unittest
import subprocess
import shutil
import chaospy as cp
import uncertainpy


folder = os.path.dirname(os.path.realpath(__file__))

test_data_dir = os.path.join(folder, "data")

seed = 10


def generate_data_allParameters():
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = uncertainpy.Parameters(parameterlist)
    model = uncertainpy.models.TestingModel1d(parameters)
    model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)



    test = uncertainpy.UncertaintyEstimation(model,
                                             features=uncertainpy.TestingFeatures(),
                                             feature_list="all",
                                             output_dir_data=test_data_dir,
                                             output_dir_figures=test_data_dir,
                                             verbose_level="error",
                                             seed=seed)

    test.allParameters()

def generate_data_singleParameters():
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = uncertainpy.Parameters(parameterlist)
    model = uncertainpy.models.TestingModel1d(parameters)
    model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)



    test = uncertainpy.UncertaintyEstimation(model,
                                             features=uncertainpy.TestingFeatures(),
                                             feature_list="all",
                                             output_dir_data=test_data_dir,
                                             output_dir_figures=test_data_dir,
                                             verbose_level="error",
                                             seed=seed)


    test.singleParameters()


def generate_data_allParametersMC():
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = uncertainpy.Parameters(parameterlist)
    model = uncertainpy.models.TestingModel1d(parameters)
    model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)



    test = uncertainpy.UncertaintyEstimation(model,
                                             features=uncertainpy.TestingFeatures(),
                                             feature_list="all",
                                             output_dir_data=test_data_dir,
                                             output_dir_figures=test_data_dir,
                                             output_data_filename="TestingModel1d_MC",
                                             verbose_level="error",
                                             seed=seed,
                                             nr_mc_samples=10**1)

    test.allParametersMC()

def generate_data_singleParametersMC():
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = uncertainpy.Parameters(parameterlist)
    model = uncertainpy.models.TestingModel1d(parameters)
    model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)



    test = uncertainpy.UncertaintyEstimation(model,
                                             features=uncertainpy.TestingFeatures(),
                                             feature_list="all",
                                             output_dir_data=test_data_dir,
                                             output_dir_figures=test_data_dir,
                                             output_data_filename="TestingModel1d_MC",
                                             verbose_level="error",
                                             seed=seed,
                                             nr_mc_samples=10**1)


    test.singleParametersMC()



def generate_data_compareMC():
    parameterlist = [["a", 1, None],
                     ["b", 2, None]]

    parameters = uncertainpy.Parameters(parameterlist)
    model = uncertainpy.TestingModel1d(parameters)
    model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


    uncertainty = uncertainpy.UncertaintyEstimations(model,
                                                     features=uncertainpy.TestingFeatures(),
                                                     feature_list="all",
                                                     verbose_level="error",
                                                     output_dir_data=test_data_dir,
                                                     output_dir_figures=test_data_dir,
                                                     nr_mc_samples=10**1,
                                                     seed=seed)


    mc_samples = [10, 100]
    uncertainty.compareMC(mc_samples)



if __name__ == "__main__":
    generate_data_allParameters()
    generate_data_singleParameters()
    generate_data_allParametersMC()
    generate_data_singleParametersMC()
    generate_data_compareMC()
