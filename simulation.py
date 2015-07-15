import os, string, cPickle, sys
import numpy as np

class Simulation():
    def __init__(self, parameterfile, modelfile, modelpath, cvode_active = True):
        self.parameterfile = parameterfile
        self.modelfile = modelfile
        self.modelpath = modelpath
        self.cvode_active = cvode_active
        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(self.filepath)

        os.chdir(self.modelpath)
        
        import neuron
        self.h = neuron.h
        self.h.load_file(1, self.modelfile)

        os.chdir(self.filedir)

        if cvode_active:
            self.h("cvode_active(1)")
        else:
            self.h("cvode_active(0)")

            
        ### Be really careful with these. Need to make sure that all references to neuron are isnide this class
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


if __name__ == "__main__":
    parameterfile = str(sys.argv[1])
    modelfile = str(sys.argv[2])
    modelpath = str(sys.argv[3])
    cvode_active = str(sys.argv[4])
    
    sim = Simulation(parameterfile, modelfile, modelpath, cvode_active)
    
    sim.runSimulation()

    V = sim.getV()
    t = sim.getT()

    tmp_V = open("tmp_V.p", "w")
    tmp_t = open("tmp_t.p", "w")
        
    cPickle.dump(V, tmp_V)
    cPickle.dump(t, tmp_t)

    tmp_V.close()
    tmp_t.close()
