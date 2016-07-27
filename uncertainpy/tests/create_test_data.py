import uncertainpy


parameterlist = [["a", 1, None],
                 ["b", 2, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.models.TestingModel1d(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)



test = uncertainpy.UncertaintyEstimation(model,
                                         CPUs=1,
                                         features=uncertainpy.TestingFeatures(),
                                         feature_list=["feature0d",
                                                       "feature1d",
                                                       "feature2d"],
                                         output_dir_data="test_data",
                                         output_dir_figures="test_data")

test.allParameters()
