import uncertainpy


# parameterlist = [["a", 0.02, None],
#                  ["b", 0.2, None],
#                  ["c", -65, None],
#                  ["d", 8, None]]
#
#
# parameters = uncertainpy.Parameters(parameterlist)
# model = uncertainpy.models.IzhikevichModel(parameters)
# model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)
#
#
# uncertainty = uncertainpy.UncertaintyEstimation(model,
#                                                 CPUs=1,
#                                                 save_figures=True,
#                                                 feature_list="all",
#                                                 output_dir_data="data/izhikevich",
#                                                 output_dir_figures="figures/izhikevich",
#                                                 nr_mc_samples=10**1,
#                                                 combined_features=True,
#                                                 verbose_level="error")
#
#
# uncertainty.singleParameters()
# uncertainty.allParameters()
