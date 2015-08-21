import os
import sys
import numpy as np

class Simulation():
    def __init__(self, parameterfile, modelfile, modelpath):
        self.parameterfile = parameterfile
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
        self.h.finitialize()
        self.h.run()


    def getT(self):
        return self.toArray(self.t)


    def getV(self):
        return self.toArray(self.V)


    def runSimulation(self):
        self.recordT()
        self.recordV()

        self.run()

        self.U = sim.getV()
        self.t = sim.getT()

    def save(self):
        np.save("tmp_U", self.U)
        np.save("tmp_t", self.t)


if __name__ == "__main__":
    parameterfile = str(sys.argv[1])
    modelfile = str(sys.argv[2])
    modelpath = str(sys.argv[3])

    sim = Simulation(parameterfile, modelfile, modelpath)
    sim.runSimulation()
    sim.save()
