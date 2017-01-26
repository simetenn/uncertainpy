from tqdm import tqdm
from xvfbwrapper import Xvfb

import numpy as np
import multiprocessing as mp

from uncertainpy.evaluateNodeFunction import evaluateNodeFunction
from uncertainpy import Data
from uncertainpy.features import GeneralFeatures

class RunModel:
    def __init__(self,
                 model,
                 features=None,
                 CPUs=mp.cpu_count(),
                 supress_model_output=True,
                 supress_model_graphics=True):


        self.model = model

        if features is None:
            self.features = GeneralFeatures(features_to_run=None)
        else:
            self.features = features

        self.CPUs = CPUs

        self.supress_model_graphics = supress_model_graphics
        self.supress_model_output = supress_model_output

        self.data = Data()


    def performInterpolation(self, ts, interpolation):
        lengths = []
        for s in ts:
            lengths.append(len(s))

        index_max_len = np.argmax(lengths)
        t = ts[index_max_len]

        interpolated_solves = []
        for inter in interpolation:
            interpolated_solves.append(inter(t))

        interpolated_solves = np.array(interpolated_solves)

        return t, interpolated_solves




    def storeResults(self, solves):

        self.data.setFeatures(solves[0])


        if self.isSolvesAdaptive(solves) and not self.model.adaptive_model:
            # TODO if the model is adaptive perform the complete interpolation here instead.
            raise ValueError("The number of simulation points varies between simulations. Try setting adaptive_model=True in model()")



        for feature in self.data.features_2d:
            if self.model.adaptive_model and feature == "directComparison":
                raise NotImplementedError("Support for >= 2d interpolation is not yet implemented")

            else:
                self.data.t[feature] = solves[0][feature][0]

                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature][1])

                # self.U[feature] = np.array(self.U[feature])

        for feature in self.data.features_1d:
            if self.model.adaptive_model and feature == "directComparison":
                ts = []
                interpolation = []
                for solved in solves:
                    ts.append(solved[feature][0])
                    interpolation.append(solved[feature][2])

                self.data.t[feature], self.data.U[feature] = self.performInterpolation(ts, interpolation)
            else:
                self.data.t[feature] = solves[0][feature][0]
                self.data.U[feature] = []
                for solved in solves:
                    self.data.U[feature].append(solved[feature][1])

                # self.data.U[feature] = np.array(self.U[feature])


        for feature in self.data.features_0d:
            self.data.U[feature] = []
            self.data.t[feature] = None
            for solved in solves:
                self.data.U[feature].append(solved[feature][1])

            # self.U[feature] = np.array(self.U[feature])

        # self.t[feature] = np.array(self.t[feature])
        self.data.U[feature] = np.array(self.data.U[feature])

        self.data.removeOnlyInvalidResults()

    def evaluateNodeFunctionList(self, nodes):
        data_list = []

        for node in nodes:
            data_list.append((self.model.cmd(),
                              self.supress_model_output,
                              self.model.adaptive_model,
                              node,
                              self.data.uncertain_parameters,
                              self.features.cmd(),
                              self.features.kwargs()))
        return data_list



    def evaluateNodes(self, nodes):

        if self.supress_model_graphics:
            vdisplay = Xvfb()
            vdisplay.start()

        solves = []
        pool = mp.Pool(processes=self.CPUs)

        for result in tqdm(pool.imap(evaluateNodeFunction,
                                     self.evaluateNodeFunctionList(nodes.T)),
                           desc="Running model",
                           total=len(nodes.T)):


            solves.append(result)


        pool.close()

        if self.supress_model_graphics:
            vdisplay.stop()

        return np.array(solves)

    # TODO should this check one specific feature.
    # Return false for all features?
    def isSolvesAdaptive(self, solves):
        """
Test if solves is an adaptive result
        """
        for feature in self.data.features_1d + self.data.features_2d:
            u_prev = solves[0][feature][1]
            for solve in solves[1:]:
                u = solve[feature][1]
                if u_prev.shape != u.shape:
                    return True
                u_prev = u
        return False


    def run(self, nodes, uncertain_parameters):

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        self.data.uncertain_parameters = uncertain_parameters

        solves = self.evaluateNodes(nodes)
        self.storeResults(solves)

        return self.data
