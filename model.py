import os
import sys
import argparse
import numpy as np

class Model():
    """
    The model must be able to handle these calls

    simulation = model()
    simulation.load()
    simulation.setParameters(parameters -> dictionary)
    simulation.run()
    simulation.save(current_process -> int)

    simulation.cmd()
    """
    def __init__(self):
        self.U = None
        self.t = None


    def load(self):
        raise NotImplementedError("No load() function implemented")


    def setParameters(self, parameters):
        raise NotImplementedError("No setParameters() function implemented")


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


    def cmd(self, additional_cmds):
        cmd = ["python", "run_model.py", "--model_name", self.__class__.__name__]
        cmd = cmd + additional_cmds

        return cmd
