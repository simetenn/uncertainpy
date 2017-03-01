import numpy as np
import os
import sys

from uncertainpy import Parameters

class Model():
    """
The model must be able to handle these calls

simulation.cmd()

simulation = model() -> __init__ must be able to run with no arguments
simulation.set_properties(properties set at runtime -> dict)
simulation.load()
simulation.setParameterValues(parameters -> dictionary)
simulation.preprocess()
simulation.run()
simulation.preprocess()
simulation.save(current_process -> int)


If you create your own model it must either be in it's own file
or the main part of the program must be inside

if __name__ == "__main__":
    # Main part of the program here

Run must store the results from the simulation in self.t and self.U
    """


    def __init__(self, parameters=None, adaptive_model=False, xlabel="", ylabel=""):
        """

----------
Required arguments

parameters: Parameters object | list of Parameter objects | list [[name, value, distribution],...]


    On the form:
    parameters = Parameters Object
    or
    parameters = [[name1, value1, distribution1],
                  [name2, value2, distribution2],
                     ...]
        name: str
            Name of the parameter
        value: number
            Value of the parameter
        distribution: None | Chaospy distribution | Function that returns a Chaospy distribution
            The distribution of the parameter.
            A parameter is considered uncertain if if has a distributiona associated
            with it.

    or
    parameters = [ParameterObject1, ParameterObject2,...]
        """
        self.U = None
        self.t = None



        self.adaptive_model = adaptive_model

        self.xlabel = xlabel
        self.ylabel = ylabel

        self.additional_cmds = []

        if isinstance(parameters, Parameters) or parameters is None:
            self.parameters = parameters
        elif isinstance(parameters, list):
            self.parameters = Parameters(parameters)
        else:
            raise ValueError("parameter argument has wrong type")


    def set_properties(self, cmds):
        self.additional_cmds = cmds.keys()
        for cmd in self.additional_cmds:
            if hasattr(self, cmd):
                raise RuntimeWarning("{} already have attribute {}".format(self.__class__.__name__, cmd))

            setattr(self, cmd, cmds[cmd])


    def reset_properties(self):
        for cmd in self.additional_cmds:
            delattr(self, cmd)

        self.additional_cmds = []


    def load(self):
        pass

    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def setDistribution(self, parameter_name, distribution_function):
        if self.parameters is None:
            raise AttributeError("Parameters is not in the model")

        self.parameters.setDistribution(parameter_name, distribution_function)


    def setAllDistributions(self, distribution_function):
        if self.parameters is None:
            raise AttributeError("Parameters are not in the model")

        self.parameters.setAllDistributions(distribution_function)

    # TODO This should raise a NotImplementedError, to make sure that users change it
    def setParameterValues(self, parameters):
        """
        Parameters: dictionary with all parameters
        """
        # How the parameters are set
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])


    def run(self):
        """
        Run must store the results from the simulation in self.t and self.U
        """
        raise NotImplementedError("No run() method implemented")


    def save(self, CPU=None, save_path=""):
        if self.U is None:
            raise ValueError("U has not been calculated")

        if self.t is None:
            self.t = np.nan

        if CPU is None:
            np.save(os.path.join(save_path, ".tmp_U"), self.U)
            np.save(os.path.join(save_path, ".tmp_t"), self.t)

        else:
            np.save(os.path.join(save_path, ".tmp_U_%d" % CPU), self.U)
            np.save(os.path.join(save_path, ".tmp_t_%d" % CPU), self.t)


    def cmd(self):
        original_path = os.path.abspath(__file__)
        original_dir = os.path.dirname(original_path)

        filepath = sys.modules[self.__class__.__module__].__file__
        filedir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if self.__class__.__module__ == "__main__":
            filedir = os.path.dirname(os.path.abspath(filename))


        cmds = [sys.executable, original_dir + "/run_model.py",
                "--model_name", self.__class__.__name__,
                "--file_dir", filedir,
                "--file_name", filename,
                "--simulation_kwargs"]

        for cmd in self.additional_cmds:
            cmds += [cmd, getattr(self, cmd)]

        return cmds
