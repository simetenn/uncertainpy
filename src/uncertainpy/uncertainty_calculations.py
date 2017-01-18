import chaospy as cp
import numpy as np


class UncertaintyCalculations:
    def __init__(self,
                 rosenblatt=False,
                 M=3,
                 nr_pc_samples=None,
                 seed=None):

        self.P = None
        self.M = M
        self.distribution = None

        self.rosenblatt = rosenblatt

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 2

        if self.rosenblatt:
            self.nr_pc_samples -= 1

        if seed is not None:
            cp.seed(seed)
            np.random.seed(seed)


    def createDistribution(self, parameter_space):
        ""


    def calculatePC(self, nodes, U):
        U_hat = cp.fit_regression(self.P, nodes, U, rule="T")
        return U_hat

    def createPCE(self):

    def createPCERosenblatt(self):

    def isAdaptiveError(self):


    def createMask(self, nodes, feature):


    def totalSensitivity(self, sensitivity="sensitivity_1"):



    def calculateNodes(self):
        return nodes

    def getNrSamples(self):
        return self.M

    def MCanalysis(self):

    def PCAnalysis(self):
