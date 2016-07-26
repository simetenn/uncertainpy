import os

from uncertainpy.plotting.plotUncertainty import PlotUncertainty

def generate_plots():
    folder = os.path.dirname(os.path.realpath(__file__))

    test_data_dir = os.path.join(folder, "../tests/data")
    data_file = "test_plot_data"
    output_test_dir = os.path.join(folder, "../tests/data")


    plot = PlotUncertainty(data_dir=test_data_dir,
                           output_dir_figures=output_test_dir,
                           output_dir_gif=output_test_dir,
                           verbose_level="error")

    plot.loadData(data_file)

    plot.plotAllData()
    plot.plot0dFeatures()

if __name__ == "__main__":
    generate_plots()
