import os

import chaospy as cp

from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy import UncertaintyEstimations, TestingFeatures, TestingModel1d
from uncertainpy import Parameters

def generate_plots_plotUncertainty():
    folder = os.path.dirname(os.path.realpath(__file__))

    data_file = "test_plot_data"
    test_data_dir = os.path.join(folder, "data")
    output_test_dir = os.path.join(folder, "data")
    # output_test_dir = os.path.join(folder, "../../test_data")


    plot = PlotUncertainty(data_dir=test_data_dir,
                           output_dir_figures=output_test_dir,
                           output_dir_gif=output_test_dir,
                           verbose_level="error")

    plot.loadData(data_file)

    plot.plotAllData()



def generate_plots_UncertaintyEstimations():
    folder = os.path.dirname(os.path.realpath(__file__))

    output_test_dir = os.path.join(folder, "../tests/data")


    def mock_distribution(x):
        return cp.Uniform(0, 1)

    parameterlist = [["a", 1, mock_distribution],
                     ["b", 2, mock_distribution]]

    parameters = Parameters(parameterlist)
    model = TestingModel1d(parameters)

    uncertainty = UncertaintyEstimations(model,
                                         features=TestingFeatures(),
                                         feature_list="all",
                                         verbose_level="error",
                                         output_dir_data=output_test_dir,
                                         output_dir_figures=output_test_dir,
                                         nr_mc_samples=10**1,
                                         seed=10)

    uncertainty.compareMC()




if __name__ == "__main__":
    generate_plots_plotUncertainty()
    # generate_plots_UncertaintyEstimations()
