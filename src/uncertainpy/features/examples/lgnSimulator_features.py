from uncertainpy.features import GeneralFeatures
import numpy as np

class LgnSimulatorFeatures(GeneralFeatures):
    def irfmax(self):
        return self.U[0]

    def irfmin(self):
        return self.U[1:].min()

    def irf_size(self):
        return self.t[np.argmin(self.U[1:])]
