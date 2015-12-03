import argparse
import os

import numpy as np

from model import Model


class NeuronModel(Model):
    def __init__(self, model_file=None, model_path=None, parameters=None):
        Model.__init__(self, parameters=parameters)

        if model_file is None or model_path is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--model_file")
            parser.add_argument("--model_path")

            args, parameter_args = parser.parse_known_args()

            self.model_file = args.model_file
            self.model_path = args.model_path
        else:
            self.model_file = model_file
            self.model_path = model_path

        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(self.filepath)

        self.h = None

    def load(self):
        os.chdir(self.model_path)

        import neuron
        self.h = neuron.h
        self.h.load_file(1, self.model_file)

        os.chdir(self.filedir)



    ### Be really careful with these. Need to make sure that all references to
    ### neuron are inside this class
    def record(self, ref_data):
        data = self.h.Vector()
        data.record(getattr(self.h, ref_data))
        return data


    def toArray(self, hocObject):
        array = np.zeros(hocObject.size())
        hocObject.to_python(array)
        return array


    def recordV(self):
        for sec in self.h.allsec():
            self.V = self.h.Vector()
            self.V.record(sec(0.5)._ref_v)
            break


    def recordT(self):
        self.t = self.record("_ref_t")


    def run(self):
        self.recordT()
        self.recordV()

        self.h.run()

        self.U = self.getV()
        self.t = self.getT()



    def getT(self):
        return self.toArray(self.t)


    def getV(self):
        return self.toArray(self.V)


    def setParameters(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))


    def cmd(self):
        additional_cmds = ["--model_file", self.model_file, "--model_path", self.model_path]
        return Model.cmd(self, additional_cmds)
