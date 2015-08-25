### TODO
# Check if cvode is active, if it is inactive avopid doing the interpolation
# Currently no way of testing if cvode is active. One way is to test if the amount of
# numbers between two different simulations are different

# Move parameter and parameter_string to uncertainty

import os
import subprocess
import time
import sys
from xvfbwrapper import Xvfb

class Model():
    def __init__(self, modelfile, modelpath, parameterfile, parameters,
                 memory_report=None, supress_output=True):
        """
        modelfile: Name of the modelfile
        modelpath: Path to the modelfile
        parameterfile: Name of file containing parameteres
        """


        self.modelfile = modelfile
        self.modelpath = modelpath
        self.parameterfile = parameterfile
        self.parameters = parameters

        self.memory_report = memory_report
        self.supress_output = supress_output

        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(self.filepath)

        self.memory_threshold = 90
        self.delta_poll = 1

        if supress_output:
            self.vdisplay = Xvfb()
            self.vdisplay.start()


    def __del__(self):
        if self.supress_output:
            self.vdisplay.stop()


    def saveParameters(self, new_parameters):
        if os.path.samefile(os.getcwd(), os.path.join(self.filedir, self.modelpath)):
            f = open(self.parameterfile, "w")
        else:
            f = open(self.modelpath + self.parameterfile, "w")

            ### This is where i am
        for parameter in self.parameters.get("name"):
            if parameter in new_parameters:
                save_parameter = new_parameters[parameter]
            else:
                save_parameter = self.parameters.get(parameter).value

            f.write(parameter + " = " + str(save_parameter) + "\n")

        f.close()


    def run(self):
        cmd = ["python", "simulation.py", self.parameterfile, self.modelfile, self.modelpath]
        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Note this checks total memory used by all applications
        if self.memory_report:
            self.memory_report.save()
            self.memory_report.saveAll()
            if self.memory_report.totalPercent() > self.memory_threshold:
                print "\nWARNING: memory threshold exceeded, aborting simulation"
                simulation.terminate()
                return -1

            time.sleep(self.delta_poll)

        ut, err = simulation.communicate()


        if simulation.returncode != 0:
            print "Error when running simulation:"
            print err
            sys.exit(1)


    def getParameters(self, fitted_parameters=None):
        if fitted_parameters:
            return self.parameters[fitted_parameters]
        else:
            return self.parameters
