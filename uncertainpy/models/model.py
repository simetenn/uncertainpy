import numpy as np
import os

class Model():
    """
    The model must be able to handle these calls

    #Note __init__ must be able to run with no arguments
    simulation = model()


    simulation.load()
    simulation.setParameters(parameters -> dictionary)
    simulation.run()
    simulation.save(current_process -> int)

    simulation.cmd()
    """
    def __init__(self, parameters=None):
        self.U = None
        self.t = None

        self.parameters = parameters


    def load(self):
        pass


    def setAllDistributions(self, distribution_function):
        if self.parameters is None:
            raise NotImplementedError("Parameters is not implemented in the model")

        self.parameters.setAllDistributions(distribution_function)


    def setDistribution(self, parameter_name, distribution_function):
        if self.parameters is None:
            raise NotImplementedError("Parameters is not implemented in the model")

        self.parameters.setDistribution(parameter_name, distribution_function)



    # def setParameters(self, parameters):
    #     raise NotImplementedError("No setParameters() function implemented")

    def setParameters(self, parameters):
        """
        Parameters: dictionary with all parameters
        """
        # How the parameters are set
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

    # def setClassParameters(self):
    #     for parameter in self.parameters:
    #         setattr(self, parameter, self.parameters[parameter].value)


    def setParameters(self, parameters):
        """
        Parameters: dictionary with all parameters
        """
        # How the parameters are set
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

    def run(self):
        raise NotImplementedError("No run() function implemented")


    def save(self, CPU=None):
        if self.t is None or self.U is None:
            raise ValueError("t or U has not been calculated")

        if CPU is None:
            np.save("tmp_U", self.U)
            np.save("tmp_t", self.t)

        else:
            np.save("tmp_U_%d" % CPU, self.U)
            np.save("tmp_t_%d" % CPU, self.t)


    def cmd(self, additional_cmds=[]):
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = ["python", filedir + "/run_model.py", "--model_name", self.__class__.__name__]
        cmd = cmd + additional_cmds

        return cmd
