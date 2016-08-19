import numpy as np
import unittest
import scipy.interpolate
import chaospy as cp
import os
import shutil
import subprocess
import glob

from uncertainpy import UncertaintyEstimation
from uncertainpy.features import TestingFeatures, NeuronFeatures
from uncertainpy.models import TestingModel0d, TestingModel1d, TestingModel2d
from uncertainpy.parameters import Parameters
from uncertainpy import Distribution
import uncertainpy


class TestUseCases(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        self.folder = os.path.dirname(os.path.realpath(__file__))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_CoffeeCupPointModelExploreParameters(self):
        parameterlist = [["kappa", -0.05, None],
                         ["u_env", 20, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.CoffeeCupPointModel(parameters)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2)


        percentages = [0.1, 0.2, 0.3]
        test_distributions = {"uniform": percentages}
        exploration.exploreParameters(test_distributions)



    def test_CoffeeCupPointModelCompareMC(self):
        parameterlist = [["kappa", -0.05, None],
                         ["u_env", 20, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.CoffeeCupPointModel(parameters)
        model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2)


        mc_samples = [10, 100]
        exploration.compareMC(mc_samples)


    def test_HodkinHuxleyModelExploreParameters(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.HodkinHuxleyModel(parameters)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2)


        percentages = [0.1, 0.2, 0.3]
        test_distributions = {"uniform": percentages}
        exploration.exploreParameters(test_distributions)



    def test_HodkinHuxleyModelCompareMC(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.HodkinHuxleyModel(parameters)
        model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2)


        mc_samples = [10, 100]
        exploration.compareMC(mc_samples)


    def test_IzhikevichModelExploreParameters(self):
        parameterlist = [["a", 0.02, None],
                         ["b", 0.2, None],
                         ["c", -65, None],
                         ["d", 8, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.IzhikevichModel(parameters)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2)


        percentages = [0.1, 0.2, 0.3]
        test_distributions = {"uniform": percentages}
        exploration.exploreParameters(test_distributions)



    def test_IzhikevichModelCompareMC(self):
        parameterlist = [["a", 0.02, None],
                         ["b", 0.2, None],
                         ["c", -65, None],
                         ["d", 8, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.IzhikevichModel(parameters)
        model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2)


        mc_samples = [10, 100]
        exploration.compareMC(mc_samples)



    # def test_LgnExploreParameters(self):
    #     model_file = "INmodel.hoc"
    #     model_path = "../models/neuron_models/dLGN_modelDB/"
    #
    #     full_model_path = os.path.join(self.folder, model_path)
    #
    #     parameterlist = [["cap", 1.1, None],
    #                      ["Rm", 22000, None],
    #                      ["Vrest", -63, None],
    #                      ["Epas", -67, None],
    #                      ["gna", 0.09, None],
    #                      ["nash", -52.6, None],
    #                      ["gkdr", 0.37, None],
    #                      ["kdrsh", -51.2, None],
    #                      ["gahp", 6.4e-5, None],
    #                      ["gcat", 1.17e-5, None]]
    #
    #
    #     parameters = uncertainpy.Parameters(parameterlist)
    #     model = uncertainpy.NeuronModel(parameters=parameters, model_file=model_file,
    #                                     model_path=full_model_path)
    #     model.setAllDistributions(uncertainpy.Distribution(0.05).uniform)
    #
    #     exploration = uncertainpy.UncertaintyEstimations(model,
    #                                                      feature_list="all",
    #                                                      output_dir_data=self.output_test_dir,
    #                                                      output_dir_figures=self.output_test_dir,
    #                                                      nr_mc_samples=10**1,
    #                                                      nr_pc_mc_samples=10**2,
    #                                                      supress_model_output=False)
    #
    #
    #     percentages = [0.1, 0.2, 0.3]
    #     test_distributions = {"uniform": percentages}
    #     exploration.exploreParameters(test_distributions)

    #
    #
    # def test_LgnModelCompareMC(self):
    #     model_file = "INmodel.hoc"
    #     model_path = "../models/neuron_models/dLGN_modelDB/"
    #
    #     full_model_path = os.path.join(self.folder, model_path)
    #
    #     parameterlist = [["cap", 1.1, None],
    #                      ["Rm", 22000, None],
    #                      ["Vrest", -63, None],
    #                      ["Epas", -67, None],
    #                      ["gna", 0.09, None],
    #                      ["nash", -52.6, None],
    #                      ["gkdr", 0.37, None],
    #                      ["kdrsh", -51.2, None],
    #                      ["gahp", 6.4e-5, None],
    #                      ["gcat", 1.17e-5, None]]
    #
    #
    #     parameters = uncertainpy.Parameters(parameterlist)
    #     model = uncertainpy.NeuronModel(parameters=parameters, model_file=model_file,
    #                                     model_path=full_model_path)
    #     model.setAllDistributions(uncertainpy.Distribution(0.05).uniform)
    #
    #     exploration = uncertainpy.UncertaintyEstimations(model,
    #                                                      feature_list="all",
    #                                                      output_dir_data=self.output_test_dir,
    #                                                      output_dir_figures=self.output_test_dir,
    #                                                      nr_mc_samples=10**1,
    #                                                      nr_pc_mc_samples=10**2)
    #
    #
    #     mc_samples = [10, 100]
    #     exploration.compareMC(mc_samples)
    #
    #

if __name__ == "__main__":
    unittest.main()
