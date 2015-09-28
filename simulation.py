import os
import sys
import argparse
import numpy as np

class Simulation():
    def __init__(self, modelfile, modelpath):
        self.modelfile = modelfile
        self.modelpath = modelpath
        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(self.filepath)

        self.U = None
        self.t = None

        os.chdir(self.modelpath)

        import neuron
        self.h = neuron.h
        self.h.load_file(1, self.modelfile)

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
        self.h.run()


    def getT(self):
        return self.toArray(self.t)


    def getV(self):
        return self.toArray(self.V)


    def runSimulation(self):
        self.recordT()
        self.recordV()

        self.run()

        self.U = self.getV()
        self.t = self.getT()

    def save(self, CPU=None):
        if CPU is None:
            np.save("tmp_U", self.U)
            np.save("tmp_t", self.t)

        else:
            np.save("tmp_U_%d" % CPU, self.U)
            np.save("tmp_t_%d" % CPU, self.t)



    def set(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a neuron simulation")
    parser.add_argument("modelfile")
    parser.add_argument("modelpath")
    parser.add_argument("--CPU", type=int)
    args, parameter_args = parser.parse_known_args()

    if len(parameter_args) % 2 != 0:
        print "ERROR: Number of parameters does not match number"
        print "         of parametervalues sent to simulation.py"
        sys.exit(1)

    parameters = {}
    i = 0
    while i < len(parameter_args):
        parameters[parameter_args[i].strip("-")] = parameter_args[i+1]
        i += 2

    sim = Simulation(args.modelfile, args.modelpath)
    sim.set(parameters)
    sim.runSimulation()
    sim.save(args.CPU)
