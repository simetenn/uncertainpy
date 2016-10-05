import numpy as np
import os
import sys


class Model():
    """
The model must be able to handle these calls

simulation = model() -> __init__ must be able to run with no arguments
simulation.load()
simulation.setParameterValues(parameters -> dictionary)
simulation.run()
simulation.save(current_process -> int)
simulation.cmd()

If you create your own model it must either be in it's own file
or the main part of the program must be inside

if __name__ == "__main__":
    # Main part of the program here

Run must store the results from the simulation in self.t and self.U
    """
    def __init__(self, parameters=None, adaptive_model=False, **additional_cmds):
        self.U = None
        self.t = None

        self.parameters = parameters
        self.adaptive_model = adaptive_model

        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(self.filepath)

        self.xlabel = ""
        self.ylabel = ""

        self.addCmds(additional_cmds)


    def addCmds(self, cmds):
        self.additional_cmds = cmds.keys()
        for cmd in self.additional_cmds:
            if hasattr(self, cmd):
                raise RuntimeWarning("{} already have attribute {}".format(self.__class__.__name__, cmd))

            setattr(self, cmd, cmds[cmd])


    def load(self):
        pass


    def setDistribution(self, parameter_name, distribution_function):
        if self.parameters is None:
            raise AttributeError("Parameters is not in the model")

        self.parameters.setDistribution(parameter_name, distribution_function)


    def setAllDistributions(self, distribution_function):
        if self.parameters is None:
            raise AttributeError("Parameters are not in the model")

        self.parameters.setAllDistributions(distribution_function)


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


        cmds = ["python", original_dir + "/run_model.py",
                "--model_name", self.__class__.__name__,
                "--file_dir", filedir,
                "--file_name", filename,
                "--kwargs"]

        for cmd in self.additional_cmds:
            cmds += [cmd, getattr(self, cmd)]

        return cmds
