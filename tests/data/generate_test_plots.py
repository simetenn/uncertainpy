import os
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from uncertainpy.plotting.plotUncertainty import PlotUncertainty
from uncertainpy.plotting.plotUncertaintyCompare import PlotUncertaintyCompare
from prettyplot import prettyPlot
from uncertainpy.features.spikes import Spikes


def generate_plots_plotUncertainty():
    folder = os.path.dirname(os.path.realpath(__file__))

    data_file = "TestingModel1d"
    test_data_dir = folder
    output_test_dir = folder

    plot = PlotUncertainty(data_dir=test_data_dir,
                           output_dir_figures=output_test_dir,
                           verbose_level="error")

    plot.loadData(data_file)


    plot.plotAllDataSensitivity()


def generate_plots_compare():
    folder = os.path.dirname(os.path.realpath(__file__))

    data_file = "TestingModel1d"
    compare_folders = ["pc", "mc_10", "mc_100"]
    test_data_dir = folder
    output_test_dir = folder
    # output_test_dir = os.path.join(folder, "../../test_data")

    plot = PlotUncertaintyCompare(data_dir=test_data_dir,
                                  output_dir_figures=output_test_dir,
                                  verbose_level="error")


    plot.plotCompareAll(data_file, compare_folders)


def generate_simulator_plot():
    folder = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = folder
    output_test_dir = folder

    U = np.load(os.path.join(test_data_dir, "U_test.npy"))
    t = np.load(os.path.join(test_data_dir, "t_test.npy"))

    prettyPlot(t, U, xlabel="", ylabel="")

    plt.savefig(os.path.join(output_test_dir, "U.png"))


def generate_spike_plot():
    folder = os.path.dirname(os.path.realpath(__file__))
    output_test_dir = folder

    t = np.arange(0, 10)
    U = np.arange(0, 10) + 10

    prettyPlot(t, U, title="Spike",
               xlabel="time", ylabel="voltage")


    plt.savefig(os.path.join(output_test_dir, "spike.png"))


def generate_spikes_plot():
    folder = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = folder
    output_test_dir = folder

    U = np.load(os.path.join(test_data_dir, "U_test.npy"))
    t = np.load(os.path.join(test_data_dir, "t_test.npy"))


    spikes = Spikes(t, U, xlabel="xlabel", ylabel="ylabel")

    spikes.plot(os.path.join(output_test_dir, "spikes.png"))

    spikes = Spikes(t, U, xlabel="xlabel", ylabel="ylabel", extended_spikes=True)

    spikes.plot(os.path.join(output_test_dir, "spikes_extended.png"))



if __name__ == "__main__":
    generate_plots_plotUncertainty()
    generate_plots_compare()
    generate_simulator_plot()
    generate_spike_plot()
    generate_spikes_plot()
