import uncertainpy

parameterlist = [["J_E", 4, None],
                 ["g", 4, None]]

parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.models.NestNetwork(parameters)
model.setAllDistributions(uncertainpy.Distribution(0.1).uniform)


exploration = uncertainpy.UncertaintyEstimation(model,
                                                CPUs=8,
                                                features=None,
                                                save_figures=True)


exploration.allParameters()
