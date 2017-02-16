from uncertainpy import UncertaintyCalculations

class TestingUncertaintyCalculations(UncertaintyCalculations):
    def PC(self, uncertain_parameters=None, method="regression", rosenblatt=False):
        arguments = {}

        arguments["function"] = "PC"
        arguments["uncertain_parameters"] = uncertain_parameters
        arguments["method"] = method
        arguments["rosenblatt"] = rosenblatt

        return arguments


    def MC(self, uncertain_parameters=None):
        arguments = {}

        arguments["function"] = "MC"
        arguments["uncertain_parameters"] = uncertain_parameters

        return arguments



    def CustomUQ(self, custom_keyword="custom_value"):
        arguments = {}

        arguments["function"] = "CustomUQ"
        arguments["custom_keyword"] = custom_keyword

        return arguments
