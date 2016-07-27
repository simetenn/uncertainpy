import os
import unittest
import subprocess
import shutil


from uncertainpy import UncertaintyEstimations, Parameters
from uncertainpy.features import TestingFeatures
from uncertainpy.models import TestingModel1d


class TestPlotUncertainpy(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.test_data_dir = os.path.join(self.folder, "data")
        self.data_file = "test_plot_data"
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        percentages = [0.01, 0.03, 0.05]

        self.test_distributions = {"uniform": percentages}


        def mock_distribution(x):
            return x

        parameterlist = [["a", 1, mock_distribution],
                         ["b", 2, mock_distribution]]

        parameters = Parameters(parameterlist)
        model = TestingModel1d(parameters)

        self.uncertainty = UncertaintyEstimations(model,
                                                  features=TestingFeatures(),
                                                  feature_list="all",
                                                  verbose_level="error",
                                                  output_dir_data=self.output_test_dir,
                                                  output_dir_figures=self.output_test_dir,
                                                  nr_mc_samples=10**1)
    #
    # def tearDown(self):
    #     if os.path.isdir(self.output_test_dir):
    #         shutil.rmtree(self.output_test_dir)


    def test_init(self):
        self.uncertainty = UncertaintyEstimations(TestingModel1d(),
                                                  features=TestingFeatures(),
                                                  feature_list="all",
                                                  verbose_level="error",
                                                  output_dir_data=self.output_test_dir,
                                                  output_dir_figures=self.output_test_dir,
                                                  nr_mc_samples=10**1)



        self.assertIsInstance(self.uncertainty, UncertaintyEstimations)


    def test_exploreParameters(self):
        self.uncertainty.exploreParameters(self.test_distributions)



if __name__ == "__main__":
    unittest.main()
