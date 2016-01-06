from uncertainpy.spikes import Spikes
import scipy.interpolate
import scipy.optimize

class Features:
    def __init__(self, t=None, U=None):
        self.spikes = None
        self.t = t
        self.U = U

        # self.implemented_features = ["nrSpikes", "timeBeforeFirstSpike"]
        self.utility_methods = ["calculateFeature",
                                "calculateFeatures",
                                "calculateAllFeatures",
                                "__init__",
                                "implementedFeatures",
                                "setSpikes",
                                "calculateSpikes"]

        if self.t is not None and self.U is not None:
            self.calculateSpikes()


    def calculateSpikes(self):
        if self.t is None:
            raise AttributeError("t is not assigned")
        if self.U is None:
            raise AttributeError("V is not assigned")

        self.spikes = Spikes()
        self.spikes.detectSpikes(self.t, self.U, thresh="auto")
        # self.spikes.plot()

    def nrSpikes(self):
        return self.spikes.nr_spikes


    def timeBeforeFirstSpike(self):
        return self.spikes.spikes[0].t_spike

    def spikeRate(self):
        return self.spikes.nr_spikes/float(self.t[-1] - self.t[0])


    def averageAPOvershoot(self):
        sum_AP_overshoot = 0
        for spike in self.spikes:
            sum_AP_overshoot += spike.U_spike
        return sum_AP_overshoot/float(self.spikes.nr_spikes)


    def averageAHPDepth(self):
        sum_AHP_depth = 0
        for i in xrange(self.spikes.nr_spikes-1):
            sum_AHP_depth += min(self.U[self.spikes[i].global_index:self.spikes[i+1].global_index])

        return sum_AHP_depth/float(self.spikes.nr_spikes)


    def averageAPWidth(self):
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
        if N == 1:
            return -1

        k = min(4, int(round(N-1)/5.))

        ISIs = []
        for i in xrange(N-1):
            ISIs.append(self.spikes[i+1].t_spike - self.spikes[i].t_spike)

        A = 0
        for i in xrange(k+1, N-1):
            A += (ISIs[i] - ISIs[i-1])/(ISIs[i] + ISIs[i-1])
        return A/(N - k - 1)


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
