import os
import numpy as np
import matplotlib.pyplot as plt

from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.plotting.prettyPlot import prettyPlot

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



def generate_plots_compare():
    folder = os.path.dirname(os.path.realpath(__file__))

    data_file = "TestingModel1d"
    compare_folders = ["pc", "mc_10", "mc_100"]
    test_data_dir = os.path.join(folder, "data")
    output_test_dir = os.path.join(folder, "data")
    # output_test_dir = os.path.join(folder, "../../test_data")

    plot = PlotUncertainty(data_dir=test_data_dir,
                           output_dir_figures=output_test_dir,
                           output_dir_gif=output_test_dir,
                           verbose_level="error")


    plot.plotCompareAll(data_file, compare_folders)


def generate_simulator_plot():
    folder = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = os.path.join(folder, "data")
    output_test_dir = os.path.join(folder, "data")

    U = np.load(os.path.join(test_data_dir, "U_test.npy"))
    t = np.load(os.path.join(test_data_dir, "t_test.npy"))

    prettyPlot(t, U, xlabel="time", ylabel="voltage")

    plt.savefig(os.path.join(output_test_dir, "U.png"))



if __name__ == "__main__":
    generate_plots_plotUncertainty()
    generate_plots_compare()
    generate_simulator_plot()
