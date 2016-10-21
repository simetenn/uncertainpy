import uncertainpy

parameterlist = [["kappa", -0.05, None],
                 ["u_env", 20, None]]


parameters = uncertainpy.Parameters(parameterlist)
model = uncertainpy.CoffeeCupPointModel(parameters)

model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


uncertainty = uncertainpy.UncertaintyEstimation(model,
                                                features=None,
                                                CPUs=1,
                                                supress_model_output=True,
                                                save_figures=True)


uncertainty.allParameters()
