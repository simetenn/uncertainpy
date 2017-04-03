from uncertainpy.features import GeneralFeatures
import numpy as np

class LgnSimulatorFeatures(GeneralFeatures):
    def irfmax(self, t, U):
        return None, U[0]

    def irfmin(self, t, U):
        return None, U[1:].min()

    def irf_size(self, t, U):
        return None, t[np.argmin(U[1:])]
