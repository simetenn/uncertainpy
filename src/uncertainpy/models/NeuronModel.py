import os

import numpy as np

from model import Model


class NeuronModel(Model):
    def __init__(self, parameters=None, adaptive_model=False,
                 model_file="mosinit.hoc", model_path=None):
        Model.__init__(self,
                       parameters=parameters,
                       adaptive_model=adaptive_model,
                       xlabel="time [ms]",
                       ylabel="voltage [mv]",
                       new_process=True)

        self.set_properties({"model_file": model_file,
                             "model_path": model_path})




    def load(self):
        current_dir = os.getcwd()
        os.chdir(self.model_path)

        import neuron

        self.h = neuron.h
        self.h.load_file(1, self.model_file)

        os.chdir(current_dir)



    ### Be really careful with these. Need to make sure that all references to
    ### neuron are inside this class
    def _record(self, ref_data):
        data = self.h.Vector()
        data.record(getattr(self.h, ref_data))
        return data


    def _toArray(self, hocObject):
        array = np.zeros(int(round(hocObject.size())))
        hocObject.to_python(array)
        return array


    def _recordV(self):
        for sec in self.h.allsec():
            self.V = self.h.Vector()
            self.V.record(sec(0.5)._ref_v)
            break


    def _recordT(self):
        self.t = self._record("_ref_t")


    def run(self, parameters):
        self.load()

        self.setParameterValues(parameters)

        self._recordT()
        self._recordV()

        self.h.run()

        self.U = self.getV()
        self.t = self.getT()

        return self.t, self.U



    def getT(self):
        return self._toArray(self.t)


    def getV(self):
        return self._toArray(self.V)


    def setParameterValues(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))
