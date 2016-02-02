import uncertainpy

plot = uncertainpy.PlotUncertainty(data_dir="data/hodgkin-huxley",
                                   output_figures_dir="figures/hodgkin-huxley")
plot.plotAllDataExploration()
