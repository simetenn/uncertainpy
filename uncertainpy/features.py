from uncertainpy.spikes import Spikes
import scipy.interpolate
import scipy.optimize
import os
import sys



class GeneralFeatures():
    def __init__(self, t=None, U=None, new_utility_methods=None):
        self.t = t
        self.U = U

        # self.implemented_features = []
        self.utility_methods = ["calculateFeature",
                                "calculateFeatures",
                                "calculateAllFeatures",
                                "__init__",
                                "implementedFeatures",
                                "cmd"]

        if new_utility_methods is None:
            new_utility_methods = []

        self.utility_methods = self.utility_methods + new_utility_methods

        self.filepath = sys.modules[self.__class__.__module__].__file__
        self.filedir = os.path.dirname(self.filepath)
        self.filename = os.path.basename(self.filepath)

        if self.__class__.__module__ == "__main__":
            self.filedir = os.path.dirname(os.path.abspath(self.filename))


    def cmd(self):
        return self.filedir, self.filename, self.__class__.__name__


    def calculateFeature(self, feature_name):
        if not callable(getattr(self, feature_name)):
            raise NotImplementedError("%s is not a implemented feature" % (feature_name))

        return getattr(self, feature_name)()


    def calculateFeatures(self, feature_names):
        results = {}
        for feature in feature_names:
            results[feature] = self.calculateFeature(feature)

        return results


    def calculateAllFeatures(self):
        results = {}
        for feature in self.implementedFeatures():
            results[feature] = self.calculateFeature(feature)

        return results


    def implementedFeatures(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods]





class NeuronFeatures(GeneralFeatures):
    def __init__(self, t=None, U=None, thresh=-30, extended_spikes=False):
        new_utility_methods = ["calculateSpikes"]

        GeneralFeatures.__init__(self, t=t, U=U, new_utility_methods=new_utility_methods)

        self.spikes = None

        if self.t is not None and self.U is not None:
            self.calculateSpikes(thresh=thresh, extended_spikes=extended_spikes)


    def calculateSpikes(self, thresh=-30, extended_spikes=False):
        if self.t is None:
            raise AttributeError("t is not assigned")
        if self.U is None:
            raise AttributeError("V is not assigned")

        self.spikes = Spikes()
        self.spikes.detectSpikes(self.t, self.U, thresh=thresh, extended_spikes=extended_spikes)





class ImplementedNeuronFeatures(NeuronFeatures):
    def __init__(self, t=None, U=None, thresh=-30, extended_spikes=False):
        NeuronFeatures.__init__(self, t=t, U=U, thresh=thresh, extended_spikes=extended_spikes)


    def nrSpikes(self):
        return self.spikes.nr_spikes


    def timeBeforeFirstSpike(self):
        if self.spikes.nr_spikes <= 0:
            return None

        return self.spikes.spikes[0].t_spike


    def spikeRate(self):
        if self.spikes.nr_spikes <= 0:
            return None

        return self.spikes.nr_spikes/float(self.t[-1] - self.t[0])


    def averageAPOvershoot(self):
        if self.spikes.nr_spikes <= 0:
            return None

        sum_AP_overshoot = 0
        for spike in self.spikes:
            sum_AP_overshoot += spike.U_spike
        return sum_AP_overshoot/float(self.spikes.nr_spikes)


    def averageAHPDepth(self):
        if self.spikes.nr_spikes <= 0:
            return None

        sum_AHP_depth = 0
        for i in xrange(self.spikes.nr_spikes-1):
            sum_AHP_depth += min(self.U[self.spikes[i].global_index:self.spikes[i+1].global_index])

        return sum_AHP_depth/float(self.spikes.nr_spikes)


    def averageAPWidth(self):
        if self.spikes.nr_spikes <= 0:
            return None

        sum_AP_width = 0
        for spike in self.spikes:
            U_width = (spike.U_spike + spike.U[0])/2.

            U_interpolation = scipy.interpolate.interp1d(spike.t, spike.U - U_width)

            # root1 = scipy.optimize.fsolve(U_interpolation, (spike.t_spike - spike.t[0])/2. + spike.t[0])
            # root2 = scipy.optimize.fsolve(U_interpolation, (spike.t[-1] - spike.t_spike)/2. + spike.t_spike)
            root1 = scipy.optimize.brentq(U_interpolation, spike.t[0], spike.t_spike)
            root2 = scipy.optimize.brentq(U_interpolation, spike.t_spike, spike.t[-1])

            sum_AP_width += abs(root2 - root1)

        return sum_AP_width/float(self.spikes.nr_spikes)


    def accomondationIndex(self):
        N = self.spikes.nr_spikes
        if N <= 1:
            return None

        k = min(4, int(round(N-1)/5.))

        ISIs = []
        for i in xrange(N-1):
            ISIs.append(self.spikes[i+1].t_spike - self.spikes[i].t_spike)

        A = 0
        for i in xrange(k+1, N-1):
            A += (ISIs[i] - ISIs[i-1])/(ISIs[i] + ISIs[i-1])
        return A/(N - k - 1)
