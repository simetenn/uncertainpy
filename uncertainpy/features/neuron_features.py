import scipy.interpolate
import scipy.optimize

from uncertainpy.features import GeneralNeuronFeatures

class NeuronFeatures(GeneralNeuronFeatures):
    def __init__(self,
                 features_to_run="all",
                 t=None,
                 U=None,
                 thresh=-30,
                 extended_spikes=False):
        
        GeneralNeuronFeatures.__init__(self,
                                       features_to_run=features_to_run,
                                       t=t, U=U,
                                       thresh=thresh,
                                       extended_spikes=extended_spikes)


    def nrSpikes(self):
        return self.spikes.nr_spikes


    def timeBeforeFirstSpike(self):
        if self.spikes.nr_spikes <= 0:
            return None

        return self.spikes.spikes[0].t_spike


    def spikeRate(self):
        if self.spikes.nr_spikes < 0:
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
