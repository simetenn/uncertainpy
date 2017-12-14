from uncertainpy.core import UncertaintyCalculations

class TestingUncertaintyCalculations(UncertaintyCalculations):
    def polynomial_chaos(self, uncertain_parameters=None, method="regression", rosenblatt=False, plot_condensed=True):
        arguments = {}

        arguments["function"] = "PC"
        arguments["uncertain_parameters"] = uncertain_parameters
        arguments["method"] = method
        arguments["rosenblatt"] = rosenblatt
        arguments["plot_condensed"] = plot_condensed

        return arguments


    def monte_carlo(self, uncertain_parameters=None, plot_condensed=True):
        arguments = {}

        arguments["function"] = "MC"
        arguments["uncertain_parameters"] = uncertain_parameters
        arguments["plot_condensed"] = plot_condensed


        return arguments



    def custom_uncertainty_quantification(self, custom_keyword="custom_value"):
        arguments = {}

        arguments["function"] = "custom_uncertainty_quantification"
        arguments["custom_keyword"] = custom_keyword

        return arguments
